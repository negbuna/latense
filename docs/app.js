// Interactive Steering Visualizer (Perfect Outer-Left Label Placement: Zero Overlap with h' Ray)
document.addEventListener('DOMContentLoaded', () => {
  const angleSlider = document.getElementById('angle-slider');
  const alphaSlider = document.getElementById('alpha-slider');
  const angleVal = document.getElementById('angle-val');
  const alphaVal = document.getElementById('alpha-val');

  const cosVal = document.getElementById('cos-val');
  const penaltyVal = document.getElementById('penalty-val');
  const magVal = document.getElementById('mag-val');
  const stateVal = document.getElementById('state-val');

  const canvas = document.getElementById('vector-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function updateVisualizer() {
    const angleDeg = parseFloat(angleSlider.value);
    const alpha = parseFloat(alphaSlider.value);

    angleVal.textContent = `${angleDeg}°`;
    alphaVal.textContent = alpha.toFixed(2);

    const angleRad = (angleDeg * Math.PI) / 180;
    const cosTheta = Math.cos(angleRad);
    const penalty = 1 - cosTheta;
    const dhMag = alpha * penalty;

    cosVal.textContent = cosTheta.toFixed(3);
    penaltyVal.textContent = penalty.toFixed(3);
    magVal.textContent = dhMag.toFixed(3);

    if (angleDeg < 45) {
      stateVal.textContent = 'Collinear';
      stateVal.style.color = '#2563eb';
    } else if (angleDeg > 120) {
      stateVal.textContent = 'Divergent';
      stateVal.style.color = '#dc2626';
    } else {
      stateVal.textContent = 'Nominal';
      stateVal.style.color = '#16a34a';
    }

    // Clear Canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Base Geometry (before dynamic fit-to-canvas scaling)
    const baseUnit = 110;
    const originX = 160;
    const originY = 230;

    // Δh (and therefore h') can extend well past the canvas at high α/θ
    // (max dhMag = 2 * 2 = 4, i.e. 440px of base unit 110). Auto-scale the
    // whole diagram uniformly each frame so nothing ever draws off-canvas,
    // while keeping every vector's relative proportions honest.
    const rightPad = 95; // reserved for arrow-tip labels
    const usableRight = canvas.width - rightPad;
    const hOffsetXUnscaled = baseUnit * Math.cos(angleRad);
    const dhOffsetXUnscaled = hOffsetXUnscaled + dhMag * baseUnit;
    const maxRightOffset = Math.max(baseUnit, dhOffsetXUnscaled, 1);
    const scale = Math.min(1, (usableRight - originX) / maxRightOffset);

    const normH = baseUnit * scale;
    const normV = baseUnit * scale;

    // Vector End Points
    const vx = originX + normV;
    const vy = originY;

    const hx = originX + normH * Math.cos(angleRad);
    const hy = originY - normH * Math.sin(angleRad);

    const dhLen = dhMag * baseUnit * scale;
    const dhx = hx + dhLen;
    const dhy = hy;

    // Grid Axes
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(originX, 20);
    ctx.lineTo(originX, originY + 25);
    ctx.moveTo(30, originY);
    ctx.lineTo(canvas.width - 30, originY);
    ctx.stroke();

    // Helper Arrow Drawer
    function drawArrow(fromX, fromY, toX, toY, color, isDashed = false) {
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2.5;

      ctx.beginPath();
      if (isDashed) {
        ctx.setLineDash([5, 4]);
      } else {
        ctx.setLineDash([]);
      }
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();
      ctx.setLineDash([]);

      const headlen = 9;
      const angle = Math.atan2(toY - fromY, toX - fromX);
      ctx.beginPath();
      ctx.moveTo(toX, toY);
      ctx.lineTo(toX - headlen * Math.cos(angle - Math.PI / 6), toY - headlen * Math.sin(angle - Math.PI / 6));
      ctx.lineTo(toX - headlen * Math.cos(angle + Math.PI / 6), toY - headlen * Math.sin(angle + Math.PI / 6));
      ctx.closePath();
      ctx.fill();
    }

    // 1. Draw Vector V (Purple - Steering Direction)
    drawArrow(originX, originY, vx, vy, '#7c3aed');
    ctx.fillStyle = '#7c3aed';
    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillText('v (Vector)', vx - 10, vy + 22);

    // 2. Draw Vector H (Blue - Original Hidden State)
    drawArrow(originX, originY, hx, hy, '#2563eb');
    // Anchor 'h (Hidden)' on the OUTER-LEFT (top-left) of vector h (opposite side of h' and v)
    const midHx = originX + 0.5 * (hx - originX);
    const midHy = originY + 0.5 * (hy - originY);
    const outerX = -22 * Math.sin(angleRad);
    const outerY = -22 * Math.cos(angleRad);
    ctx.fillStyle = '#2563eb';
    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillText('h (Hidden)', midHx + outerX - 58, midHy + outerY - 4);

    // 3. Draw Correction Δh (Green - Parallel to v)
    drawArrow(hx, hy, dhx, dhy, '#16a34a');
    ctx.fillStyle = '#16a34a';
    ctx.font = '600 13px Inter, sans-serif';
    ctx.fillText('Δh', hx + Math.max(6, dhLen / 2 - 8), hy - 12);

    // 4. Draw Resultant Vector h' = h + Δh (Dashed Teal Arrow)
    if (dhLen > 2) {
      drawArrow(originX, originY, dhx, dhy, '#0d9488', true);
      // Place label h' on the LOWER-RIGHT side of the dashed tip
      ctx.fillStyle = '#0d9488';
      ctx.font = '600 13px Inter, sans-serif';
      ctx.fillText("h' = h + Δh", dhx + 10, dhy + 16);
    }

    // 5. Draw Angle Arc (Orange - Original Angle θ)
    if (angleDeg > 3) {
      ctx.strokeStyle = '#ea580c';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(originX, originY, 36 * scale, 0, -angleRad, true);
      ctx.stroke();

      const midAngle = angleRad / 2;
      const arcTextX = originX + 48 * scale * Math.cos(midAngle);
      const arcTextY = originY - 48 * scale * Math.sin(midAngle);

      ctx.fillStyle = '#ea580c';
      ctx.font = '500 12px Inter';
      ctx.fillText(`θ = ${angleDeg}°`, arcTextX, arcTextY);
    }
  }

  angleSlider.addEventListener('input', updateVisualizer);
  alphaSlider.addEventListener('input', updateVisualizer);

  updateVisualizer();
});
