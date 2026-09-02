import torch
import json
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

def gated_steered_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    steering_vector: torch.Tensor,
    alpha: float = 0.3,
    similarity_threshold: float = None,
    max_new_tokens: int = 1024,
    device: str = "cuda"
) -> str:
    """
    Generates text using a steering vector that is 'gated' off if JSON syntax is broken.
    """
    eos_token = tokenizer.eos_token
    if eos_token not in stop_words:
        stop_words.append(eos_token)

    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    generated_ids = []
    steering_vector = torch.nn.functional.normalize(steering_vector.to(device).squeeze(), dim=-1)

    for token_idx in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            all_hidden_states = outputs.hidden_states

            num_layers = len(model.model.layers)
            target_layer_idx = num_layers - 1

            if 0 <= target_layer_idx < len(all_hidden_states):
                target_hidden_state = all_hidden_states[target_layer_idx]
                last_token_hs = target_hidden_state[:, -1, :]
                current_alpha = alpha

                # gating logic for json
                if token_idx > 5:
                    current_gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    try:
                        json_start = current_gen_text.find('{')
                        if json_start != -1:
                            _ = json.loads(current_gen_text[json_start:] + '}')
                    except json.JSONDecodeError:
                        current_alpha = 0.0

                similarity = torch.nn.functional.cosine_similarity(last_token_hs.squeeze(), steering_vector, dim=-1)

                if similarity_threshold is None or similarity < similarity_threshold:
                    nudge = (current_alpha * steering_vector).unsqueeze(0)

                    modified_hs = target_hidden_state.clone()
                    modified_hs[:, -1, :] += nudge

                    # args for the layer forward pass
                    # attention_mask needs to be updated for current input_ids
                    current_attention_mask = torch.ones(input_ids.shape, dtype=torch.bool, device=device)
                    seq_length = input_ids.shape[1]
                    position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0)

                    causal_mask = _prepare_4d_causal_attention_mask(
                        current_attention_mask, (1, seq_length), modified_hs, 0
                    )

                    # pre-calculate position embeddings once
                    position_embeddings = model.model.rotary_emb(all_hidden_states[0], position_ids)

                    temp_hs = modified_hs
                    for i in range(target_layer_idx, num_layers):
                        layer_outputs = model.model.layers[i](
                            temp_hs,
                            attention_mask=causal_mask,
                            position_ids=position_ids,
                            position_embeddings=position_embeddings,
                        )
                        temp_hs = layer_outputs[0]

                    temp_hs = model.model.norm(temp_hs)
                    # Ensure batch dimension is present
                    if temp_hs.dim() == 2:
                        temp_hs = temp_hs.unsqueeze(0) # Add batch_size dimension back
                    next_token_logits = model.lm_head(temp_hs)[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            new_token = tokenizer.decode(next_token_id.item())
            if new_token in stop_words:
                break
            
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)
