"""Render the LaTense interactive-steering demo as 1080p / 60fps PNG frames.

After running this script (frames land in ./demo_frames), encode with:

  ffmpeg -y -framerate 60 -i demo_frames/frame_%04d.png \
    -c:v libx264 -pix_fmt yuv420p -profile:v high -crf 15 -preset slow \
    -movflags +faststart latense_visualizer_demo.mp4

  ffmpeg -y -framerate 60 -i demo_frames/frame_%04d.png \
    -vf "fps=30,scale=1080:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a" \
    latense_visualizer_demo.gif
"""

import math
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc

FONT_SANS = "Helvetica Neue"
FONT_MONO = "Menlo"
plt.rcParams["font.family"] = FONT_SANS
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = FONT_SANS
plt.rcParams["mathtext.it"] = f"{FONT_SANS}:italic"
plt.rcParams["mathtext.bf"] = f"{FONT_SANS}:bold"

BG = "#080d1a"
CARD_BG = "#0f172a"
CARD_EDGE = "#1e293b"
ROW_BG = "#111c33"
TEXT_PRIMARY = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
TEXT_FAINT = "#64748b"
CYAN = "#38bdf8"
PURPLE = "#a855f7"
GREEN = "#22c55e"
TEAL = "#2dd4bf"
ORANGE = "#f97316"

STATE_STYLES = {
    "Collinear": dict(fg=CYAN, bg="#082f49", edge="#0c4a6e",
                      desc="Attenuated steering prevents over-intervention"),
    "Nominal": dict(fg=GREEN, bg="#052e17", edge="#14532d",
                    desc="Proportional restorative geometric correction"),
    "Divergent": dict(fg=ORANGE, bg="#431407", edge="#7c2d12",
                      desc="High restorative force prevents text-looping collapse"),
}

W, H = 1920, 1080
DPI = 100
FPS = 60
DURATION = 7.5
N_FRAMES = int(FPS * DURATION)
ALPHA_FIXED = 0.5

THIN_SPACE = " "


def tracked(text, gap=1):
    return (THIN_SPACE * gap).join(list(text))


def steering_state(theta_deg):
    if theta_deg < 45:
        return "Collinear"
    if theta_deg > 120:
        return "Divergent"
    return "Nominal"


# --- text measurement -------------------------------------------------------
# Data coords are 1:1 with pixels (xlim/ylim == W/H at DPI 100), so a display
# space measurement can be used directly as a width/height in layout units.
_MEASURE_FIG = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
_MEASURE_CANVAS = _MEASURE_FIG.canvas
_MEASURE_CANVAS.draw()
_MEASURE_RENDERER = _MEASURE_CANVAS.get_renderer()


def measure_text(text, fontsize, family=FONT_SANS, weight="normal"):
    t = _MEASURE_FIG.text(0, 0, text, fontsize=fontsize, family=family, fontweight=weight)
    bb = t.get_window_extent(renderer=_MEASURE_RENDERER)
    t.remove()
    return bb.width, bb.height


def ease_smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# Piecewise-linear angle path: constant angular velocity inside each leg, so
# motion is visible from frame 0 (no ease-in dead time at the loop seam), with
# the corners rounded only at the two interior turning points.
KEYFRAMES = [(0.0, 60.0), (2.5, 15.0), (5.0, 135.0), (7.5, 60.0)]
FILLET = 0.4


def _segment_value_slope(t):
    for (t0, v0), (t1, v1) in zip(KEYFRAMES[:-1], KEYFRAMES[1:]):
        if t0 <= t <= t1:
            slope = (v1 - v0) / (t1 - t0)
            return v0, slope, t0
    t0, v0 = KEYFRAMES[-1]
    return v0, 0.0, t0


def angle_at(t):
    t = t % DURATION
    corners = [tt for tt, _ in KEYFRAMES[1:-1]]
    for c in corners:
        if abs(t - c) < FILLET:
            v0_in, slope_in, t0_in = _segment_value_slope(c - 1e-6)
            v0_out, slope_out, t0_out = _segment_value_slope(c + 1e-6)
            left_ext = (v0_in + slope_in * (c - t0_in)) + slope_in * (t - c)
            right_ext = (v0_out + slope_out * (c - t0_out)) + slope_out * (t - c)
            w = ease_smoothstep((t - (c - FILLET)) / (2 * FILLET))
            return left_ext * (1 - w) + right_ext * w
    v0, slope, t0 = _segment_value_slope(t)
    return v0 + slope * (t - t0)


def rounded_card(ax, x, y, w, h, fc=CARD_BG, ec=CARD_EDGE, lw=1.4, radius=16):
    """radius is in pixels (data units), not a fraction."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=1,
        mutation_aspect=1,
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color, lw=3.2, dashed=False, zorder=5):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=zorder,
        linestyle=(0, (5, 3)) if dashed else "solid",
        shrinkA=0, shrinkB=0,
    )
    ax.add_patch(arrow)


def halo_text(ax, x, y, s, *, color, fontsize, ha, va, weight="bold",
              family=FONT_SANS, halo=CARD_BG, zorder=9):
    """Label with an opaque plate behind it so a vector can never run through it."""
    return ax.text(
        x, y, s, color=color, fontsize=fontsize, family=family,
        fontweight=weight, ha=ha, va=va, zorder=zorder,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=halo, edgecolor="none", alpha=0.92),
    )


def draw_left_panel(ax, ox, oy, pw, ph, theta_deg, alpha):
    rounded_card(ax, ox, oy, pw, ph)

    pad_x = 42
    top = oy + ph
    ax.text(ox + pad_x, top - 38, tracked("LATENT TRAJECTORY MANIFOLD"),
            fontsize=13, color=CYAN, fontweight="bold", family=FONT_SANS,
            ha="left", va="top", zorder=6)
    ax.text(ox + pad_x, top - 74, "Dynamic Geometric Space",
            fontsize=25, color=TEXT_PRIMARY, fontweight="bold", family=FONT_SANS,
            ha="left", va="top", zorder=6)

    # Plot region
    gx0, gx1 = ox + 54, ox + pw - 54
    gy0, gy1 = oy + 70, top - 165
    gw, gh = gx1 - gx0, gy1 - gy0
    origin_x = gx0 + gw * 0.42
    origin_y = gy0 + gh * 0.17
    unit = min(gw, gh) * 0.60

    theta_rad = math.radians(theta_deg)
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    dh_mag = alpha * (1 - cos_t)

    # Fit the whole construction inside the plot region, uniformly, so the
    # proportions stay honest at every angle (Δh grows to 0.85*unit at 135deg).
    label_pad = 150
    h_unit_x, h_unit_y = unit * cos_t, unit * sin_t
    reach_r = max(h_unit_x + dh_mag * unit, unit)
    reach_l = min(h_unit_x, 0.0)
    reach_t = max(h_unit_y, 0.0)
    scale = 1.0
    if reach_r > 0:
        scale = min(scale, (gx1 - label_pad - origin_x) / reach_r)
    if reach_l < 0:
        scale = min(scale, (origin_x - (gx0 + label_pad)) / -reach_l)
    if reach_t > 0:
        scale = min(scale, (gy1 - 60 - origin_y) / reach_t)
    scale = max(scale, 0.05)

    unit_s = unit * scale
    v_end = (origin_x + unit_s, origin_y)
    h_end = (origin_x + unit_s * cos_t, origin_y + unit_s * sin_t)
    dh_len = dh_mag * unit_s
    dh_end = (h_end[0] + dh_len, h_end[1])

    ax.plot([gx0, gx1], [origin_y, origin_y], color="#1b2740", lw=1, zorder=2)
    ax.plot([origin_x, origin_x], [gy0, gy1], color="#1b2740", lw=1, zorder=2)

    # v — steering vector (always along +x)
    draw_arrow(ax, origin_x, origin_y, v_end[0], v_end[1], PURPLE)
    halo_text(ax, origin_x + (v_end[0] - origin_x) * 0.62, origin_y - 28,
              "v (Steering Vector)", color=PURPLE, fontsize=13, ha="center", va="top")

    # h — hidden state. Anchor on the outward normal (theta+90deg), i.e. the
    # side away from the wedge, so it never lands on v, the arc, or h'.
    draw_arrow(ax, origin_x, origin_y, h_end[0], h_end[1], CYAN)
    nx, ny = -sin_t, cos_t
    off = 22
    mid_hx = origin_x + 0.56 * (h_end[0] - origin_x) + nx * off
    mid_hy = origin_y + 0.56 * (h_end[1] - origin_y) + ny * off
    halo_text(ax, mid_hx, mid_hy, "h (Hidden State)", color=CYAN, fontsize=13,
              ha="right" if nx <= 0 else "left", va="center")

    if dh_len > 3:
        # Delta-h — correction, drawn parallel to v from the tip of h
        draw_arrow(ax, h_end[0], h_end[1], dh_end[0], dh_end[1], GREEN)
        halo_text(ax, (h_end[0] + dh_end[0]) / 2, h_end[1] + 24,
                  r"$\Delta h = \alpha(1-\cos\theta)\,v$", color=GREEN,
                  fontsize=13, ha="center", va="bottom")

        # h' — resultant
        draw_arrow(ax, origin_x, origin_y, dh_end[0], dh_end[1], TEAL, dashed=True)
        halo_text(ax, dh_end[0] + 14, dh_end[1] - 6, r"$h' = h + \Delta h$",
                  color=TEAL, fontsize=13.5, ha="left", va="top")

    if theta_deg > 2:
        arc_r = min(gh * 0.13, unit_s * 0.42)
        ax.add_patch(Arc((origin_x, origin_y), arc_r * 2, arc_r * 2, angle=0,
                         theta1=0, theta2=theta_deg, color=ORANGE, lw=2.0, zorder=4))
        mid_ang = math.radians(theta_deg / 2)
        lr = arc_r + 30
        halo_text(ax, origin_x + lr * math.cos(mid_ang), origin_y + lr * math.sin(mid_ang),
                  rf"$\theta = {theta_deg:.0f}\degree$", color=ORANGE, fontsize=12.5,
                  ha="center", va="center", weight="normal")

    halo_text(ax, origin_x - 10, origin_y - 12, r"$h_t$", color=TEXT_FAINT,
              fontsize=13, ha="right", va="top", weight="normal")


def draw_right_panel(ax, ox, oy, pw, ph, theta_deg, alpha):
    rounded_card(ax, ox, oy, pw, ph)

    pad_x = 42
    inner_x = ox + pad_x
    inner_w = pw - 2 * pad_x
    text_inset = 24
    top = oy + ph

    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    penalty = 1 - cos_t
    dh_mag = alpha * penalty
    state = steering_state(theta_deg)
    style = STATE_STYLES[state]

    ax.text(inner_x, top - 38, tracked("REAL-TIME GOVERNOR TELEMETRY"),
            fontsize=13, color=CYAN, fontweight="bold", family=FONT_SANS,
            ha="left", va="top", zorder=6)
    ax.text(inner_x, top - 74, "Steering Dynamics & Metrics",
            fontsize=25, color=TEXT_PRIMARY, fontweight="bold", family=FONT_SANS,
            ha="left", va="top", zorder=6)

    # State pill
    pill_h = 88
    pill_y = top - 190 - pill_h
    rounded_card(ax, inner_x, pill_y, inner_w, pill_h,
                 fc=style["bg"], ec=style["edge"], lw=1.4, radius=14)
    dot_x = inner_x + text_inset + 7
    ax.scatter([dot_x], [pill_y + pill_h * 0.66], s=74, color=style["fg"], zorder=7)
    ax.text(dot_x + 20, pill_y + pill_h * 0.66, f"{state.upper()} ALIGNMENT",
            fontsize=15, color=style["fg"], fontweight="bold", family=FONT_SANS,
            ha="left", va="center", zorder=7)
    ax.text(inner_x + text_inset, pill_y + pill_h * 0.28, style["desc"],
            fontsize=12, color=TEXT_MUTED, family=FONT_SANS,
            ha="left", va="center", zorder=7)

    # Metric rows — values are right-aligned INSIDE the row box.
    metrics = [
        ("Alignment Angle (θ)", f"{theta_deg:.1f}°", ORANGE),
        ("Cosine Similarity cos(θ)", f"{cos_t:.3f}", TEXT_PRIMARY),
        ("Geometric Penalty (1 − cos θ)", f"{penalty:.3f}", ORANGE),
        ("Effective Correction ‖Δh‖", f"{dh_mag:.3f}", GREEN),
    ]
    row_h, row_gap = 70, 14
    rows_top = pill_y - 26
    for i, (label, value, vcolor) in enumerate(metrics):
        ry = rows_top - i * (row_h + row_gap) - row_h
        rounded_card(ax, inner_x, ry, inner_w, row_h, fc=ROW_BG, ec="#1e293b",
                     lw=1.0, radius=12)
        ax.text(inner_x + text_inset, ry + row_h / 2, label,
                fontsize=13, color=TEXT_MUTED, family=FONT_SANS,
                ha="left", va="center", zorder=7)
        ax.text(inner_x + inner_w - text_inset, ry + row_h / 2, value,
                fontsize=16, color=vcolor, fontweight="bold", family=FONT_MONO,
                ha="right", va="center", zorder=7)
    rows_bottom = rows_top - 4 * row_h - 3 * row_gap

    # Governor / slider block
    gov_h = 230
    gov_y = oy + 40
    if gov_y + gov_h > rows_bottom - 20:          # keep clear of the rows above
        gov_h = max(190, rows_bottom - 20 - gov_y)
    rounded_card(ax, inner_x, gov_y, inner_w, gov_h, fc="#0b1526", ec="#1e293b",
                 lw=1.2, radius=14)
    gtop = gov_y + gov_h
    ax.text(inner_x + text_inset, gtop - 26, tracked("INTERACTIVE ANGLE GOVERNOR"),
            fontsize=11.5, color=CYAN, fontweight="bold", family=FONT_SANS,
            ha="left", va="top", zorder=7)
    ax.text(inner_x + text_inset, gtop - 56, "Simulating dynamic hidden state drift (θ):",
            fontsize=12.5, color=TEXT_PRIMARY, family=FONT_SANS,
            ha="left", va="top", zorder=7)

    track_x0 = inner_x + text_inset
    track_x1 = inner_x + inner_w - text_inset
    track_y = gov_y + gov_h * 0.46
    ax.plot([track_x0, track_x1], [track_y, track_y], color="#1e293b", lw=5,
            solid_capstyle="round", zorder=6)
    fill_x = track_x0 + (track_x1 - track_x0) * (theta_deg / 180.0)
    ax.plot([track_x0, fill_x], [track_y, track_y], color=CYAN, lw=5,
            solid_capstyle="round", zorder=7)
    ax.scatter([fill_x], [track_y], s=170, color="white", edgecolor=CYAN,
               linewidth=2.4, zorder=8)

    tick_y = track_y - 22
    for tx, label, ha in ((track_x0, "0° (Collinear)", "left"),
                          ((track_x0 + track_x1) / 2, "90° (Orthogonal)", "center"),
                          (track_x1, "180° (Opposite)", "right")):
        ax.text(tx, tick_y, label, fontsize=10.5, color=TEXT_FAINT,
                family=FONT_SANS, ha=ha, va="top", zorder=7)

    ax.text((track_x0 + track_x1) / 2, gov_y + 30,
            r"$h' = h + \alpha \cdot (1-\cos\theta) \cdot \frac{\|h\|}{\|v\|} \cdot v$",
            fontsize=17, color="#cbd5e1", family=FONT_SANS,
            ha="center", va="center", zorder=7)


def draw_frame(theta_deg, alpha, out_path):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(BG)

    # Masthead
    ax.text(60, H - 62, "LaTense", fontsize=30, color=CYAN, fontweight="bold",
            family=FONT_SANS, ha="left", va="center", zorder=6)
    ax.text(232, H - 62, tracked("GEOMETRIC ALIGNMENT GOVERNOR FOR ACTIVATION STEERING"),
            fontsize=14, color=TEXT_MUTED, family=FONT_SANS,
            ha="left", va="center", zorder=6)

    # URL chip — sized from the measured text so it can never clip.
    chip_text = "nathanegbuna.com/latense"
    chip_fs = 13.5
    tw, th = measure_text(chip_text, chip_fs, family=FONT_MONO)
    chip_w, chip_h = tw + 44, th + 24
    chip_x, chip_y = W - 60 - chip_w, H - 62 - chip_h / 2
    rounded_card(ax, chip_x, chip_y, chip_w, chip_h, fc="#0f172a", ec="#334155",
                 lw=1.2, radius=chip_h / 2)
    ax.text(chip_x + chip_w / 2, chip_y + chip_h / 2, chip_text,
            fontsize=chip_fs, color="#cbd5e1", family=FONT_MONO,
            ha="center", va="center", zorder=6)

    panel_bottom, panel_top = 40, H - 130
    panel_h = panel_top - panel_bottom
    gap = 40
    panel_w = (W - 80 - gap) / 2
    left_x = 40
    right_x = left_x + panel_w + gap

    draw_left_panel(ax, left_x, panel_bottom, panel_w, panel_h, theta_deg, alpha)
    draw_right_panel(ax, right_x, panel_bottom, panel_w, panel_h, theta_deg, alpha)

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_frames")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for i in range(N_FRAMES):
        t = i / FPS
        theta = angle_at(t)
        draw_frame(theta, ALPHA_FIXED, os.path.join(out_dir, f"frame_{i:04d}.png"))
        if i % 30 == 0:
            print(f"frame {i}/{N_FRAMES}  theta={theta:.2f}")

    print(f"done — frames written to {out_dir}")


if __name__ == "__main__":
    main()
