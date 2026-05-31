#!/usr/bin/env python3
"""Generate animated GIF logo for doc_classifier_gpu README."""

from PIL import Image, ImageDraw, ImageFont
import math, os

# ── Config ────────────────────────────────────────────────────────────────────
W, H     = 800, 380
FPS      = 15
N_FRAMES = 60
OUT      = os.path.join(os.path.dirname(__file__), "..", "logo.gif")

FONT_PATH = "/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Palette
BG       = (26,  26,  46)
TEXT     = (224, 224, 224)
GREEN    = (118, 185,   0)
BLUE     = ( 68, 119, 255)
ORANGE   = (204, 102,  34)
DIM      = ( 44,  44,  80)
DIMMER   = ( 30,  30,  55)
WHITE    = (255, 255, 255)
YELLOW   = (255, 204,   0)

def font(size):
    try:    return ImageFont.truetype(FONT_PATH, size)
    except: return ImageFont.load_default()

def mono(size):
    try:    return ImageFont.truetype(FONT_MONO, size)
    except: return font(size)

# ── Corpus samples ────────────────────────────────────────────────────────────
DOCS = [
    ("fieldmanual_007.pdf",  "strategy",          88),
    ("radio_intercept.pdf",  "INFOWAR",           79),
    ("patrol_photo.jpg",     "camps",             64),
    ("tc3-23-35.pdf",        "Liberated_manuals", 91),
    ("mine_diagram.pdf",     "mines",             83),
    ("medkit_guide.pdf",     "medical",           76),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    return tuple(int(lerp(a, b, t)) for a, b in zip(c1, c2))

def ease_out(t):
    return 1 - (1 - t) ** 3

def alpha_blend(base, over, a):
    return tuple(int(b * (1 - a) + o * a) for b, o in zip(base, over))

def draw_rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill,
                            outline=outline, width=width)

def glow_text(draw, pos, text, fnt, color, layers=3):
    x, y = pos
    for i in range(layers, 0, -1):
        a = 0.15 / i
        gc = alpha_blend(BG, color, a)
        for dx in range(-i, i+1):
            for dy in range(-i, i+1):
                draw.text((x+dx, y+dy), text, font=fnt, fill=gc)
    draw.text(pos, text, font=fnt, fill=color)

# ── Frame builder ─────────────────────────────────────────────────────────────
def make_frame(f):
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    t_total = f / (N_FRAMES - 1)   # 0 → 1 over the whole animation

    # ── Background panel lines ────────────────────────────────────────────────
    for y in range(0, H, 24):
        draw.line([(0, y), (W, y)], fill=(28, 28, 50), width=1)

    # ── Left doc panel ────────────────────────────────────────────────────────
    panel_x = 18
    panel_w = 320
    draw_rounded_rect(draw, [panel_x, 60, panel_x + panel_w, H - 20],
                      6, DIMMER, outline=(50, 50, 90), width=1)

    # ── Right result panel ────────────────────────────────────────────────────
    rpanel_x = panel_x + panel_w + 30
    rpanel_w = W - rpanel_x - 18
    draw_rounded_rect(draw, [rpanel_x, 60, rpanel_x + rpanel_w, H - 20],
                      6, DIMMER, outline=(50, 50, 90), width=1)

    # ── Title ─────────────────────────────────────────────────────────────────
    title_prog = ease_out(min(1.0, t_total / 0.15))
    title_full = "doc_classifier_gpu"
    n_chars    = max(1, int(len(title_full) * title_prog))
    title_str  = title_full[:n_chars]
    cursor     = "_" if f % 6 < 3 and n_chars < len(title_full) else ""

    fnt_title  = font(22)
    fnt_sub    = mono(11)
    fnt_body   = mono(12)
    fnt_label  = mono(13)

    glow_text(draw, (panel_x, 14), title_str + cursor, fnt_title, GREEN, layers=4)

    sub = "BAAI/bge-m3  ·  EasyOCR  ·  ProcessPoolExecutor(14)  ·  LogisticRegression"
    if title_prog > 0.8:
        sub_a = ease_out((title_prog - 0.8) / 0.2)
        sc    = lerp_color(BG, (130, 130, 160), sub_a)
        draw.text((panel_x, 40), sub, font=fnt_sub, fill=sc)

    # ── Column headers ────────────────────────────────────────────────────────
    if title_prog > 0.5:
        draw.text((panel_x + 8,  64), "document", font=fnt_sub, fill=(100, 100, 140))
        draw.text((rpanel_x + 8, 64), "classification", font=fnt_sub, fill=(100, 100, 140))

    # ── Scanning beam: sweeps 16→40, then holds ───────────────────────────────
    beam_start_f, beam_end_f = 16, 40
    if f >= beam_start_f:
        beam_t    = min(1.0, (f - beam_start_f) / (beam_end_f - beam_start_f))
        beam_prog = ease_out(beam_t)
        beam_x    = int(lerp(panel_x + panel_w + 4, rpanel_x - 4, beam_prog))

        # Vertical glowing beam
        for dx in range(-4, 5):
            alpha = max(0, 1 - abs(dx) / 5.0) * 0.7
            bc    = lerp_color(BG, BLUE, alpha)
            draw.line([(beam_x + dx, 62), (beam_x + dx, H - 22)], fill=bc)
    else:
        beam_t = 0.0

    # ── Documents (left panel) ────────────────────────────────────────────────
    doc_start_f = 9
    row_h = (H - 100) // len(DOCS)
    for i, (fname, cls, pct) in enumerate(DOCS):
        appear_t = ease_out(min(1.0, max(0.0,
                    (f - doc_start_f - i * 2) / 6)))
        if appear_t <= 0:
            continue

        ry = 80 + i * row_h
        fc = lerp_color(BG, TEXT, appear_t)

        # Highlight row if beam has passed
        doc_beam_t = min(1.0, max(0.0, beam_t * len(DOCS) - i)) if f >= beam_start_f else 0.0
        if doc_beam_t > 0:
            hc = lerp_color(DIMMER, (30, 40, 65), min(1.0, doc_beam_t))
            draw.rectangle([panel_x+2, ry-2, panel_x+panel_w-2, ry+row_h-4], fill=hc)

        # Extension dot
        ext_color = ORANGE if fname.endswith(".jpg") else (80, 120, 200)
        ext = fname.rsplit(".", 1)[-1].upper()
        draw_rounded_rect(draw, [panel_x+8, ry+2, panel_x+42, ry+16],
                          3, ext_color)
        draw.text((panel_x+10, ry+3), ext[:3], font=mono(9), fill=WHITE)
        draw.text((panel_x+46, ry+2), fname[:26], font=fnt_body, fill=fc)

    # ── Results (right panel) ─────────────────────────────────────────────────
    for i, (fname, cls, pct) in enumerate(DOCS):
        # Result appears as beam passes this row
        result_t = 0.0
        if f >= beam_start_f:
            result_t = ease_out(min(1.0, max(0.0,
                        (beam_t * len(DOCS) - i + 0.3) / 0.8)))
        if result_t <= 0:
            continue

        ry = 80 + i * row_h

        # Class label
        lc = lerp_color(BG, ORANGE, result_t)
        draw.text((rpanel_x + 8, ry + 2), cls, font=fnt_label, fill=lc)

        # Confidence bar
        bar_w   = int((rpanel_w - 90) * (pct / 100) * result_t)
        bar_x   = rpanel_x + 8
        bar_y   = ry + 18
        bar_h   = 6
        bar_bg  = (40, 40, 70)
        bar_full_w = rpanel_w - 90

        draw.rectangle([bar_x, bar_y, bar_x + bar_full_w, bar_y + bar_h], fill=bar_bg)
        if bar_w > 0:
            # Green gradient bar
            for bx in range(bar_w):
                t_bar = bx / max(1, bar_full_w)
                bc = lerp_color(GREEN, (0, 200, 80), t_bar)
                draw.line([(bar_x + bx, bar_y), (bar_x + bx, bar_y + bar_h)], fill=bc)

        # Percentage
        pct_shown = int(pct * result_t)
        pc = lerp_color(BG, GREEN, result_t)
        draw.text((bar_x + bar_full_w + 6, bar_y - 2),
                  f"{pct_shown}%", font=mono(11), fill=pc)

    # ── GPU utilization bar (bottom) ──────────────────────────────────────────
    gpu_t = min(1.0, max(0.0,
            (f - beam_start_f) / (beam_end_f - beam_start_f))) if f >= beam_start_f else 0.0
    # Pulse after beam finishes
    if f > beam_end_f:
        pulse = 0.85 + 0.15 * math.sin((f - beam_end_f) * 0.4)
        gpu_t = gpu_t * pulse

    bar_y    = H - 16
    bar_lx   = panel_x
    bar_rx   = W - 18
    bar_full = bar_rx - bar_lx
    bar_fill = int(bar_full * gpu_t)

    draw.text((bar_lx, bar_y - 13), "GPU 0 (RTX 5080)", font=mono(9), fill=(100, 100, 140))
    pct_gpu  = int(gpu_t * 100)
    draw.text((bar_rx - 38, bar_y - 13), f"{pct_gpu:3d}%", font=mono(9), fill=GREEN)
    draw.rectangle([bar_lx, bar_y, bar_rx, bar_y + 5], fill=(30, 30, 60))
    if bar_fill > 0:
        for bx in range(bar_fill):
            tc = lerp_color((30, 100, 0), GREEN, bx / max(1, bar_full))
            draw.line([(bar_lx + bx, bar_y), (bar_lx + bx, bar_y + 5)], fill=tc)

    # ── "militia.joblib saved" finale ─────────────────────────────────────────
    if f >= 41:
        fin_t = ease_out(min(1.0, (f - 41) / 8))
        fc    = lerp_color(BG, YELLOW, fin_t)
        msg   = "militia.joblib  ✓  saved"
        tw    = draw.textlength(msg, font=fnt_label)
        glow_text(draw, (W // 2 - int(tw // 2), H - 16 - 26),
                  msg, fnt_label, lerp_color(BG, YELLOW, fin_t), layers=3)

    # ── Fade for loop ─────────────────────────────────────────────────────────
    if f >= 52:
        fade_a = (f - 52) / (N_FRAMES - 52)
        overlay = Image.new("RGB", (W, H), BG)
        img = Image.blend(img, overlay, fade_a * 0.85)

    return img

# ── Render ────────────────────────────────────────────────────────────────────
print(f"Rendering {N_FRAMES} frames at {W}x{H} ...")
frames = []
for f in range(N_FRAMES):
    frames.append(make_frame(f))
    if f % 10 == 0:
        print(f"  frame {f}/{N_FRAMES}")

durations = [1000 // FPS] * N_FRAMES
durations[-1] = 800  # pause on last frame before loop

frames[0].save(
    OUT,
    save_all=True,
    append_images=frames[1:],
    loop=0,
    duration=durations,
    optimize=False,
)
print(f"Saved → {OUT}")
size_kb = os.path.getsize(OUT) // 1024
print(f"File size: {size_kb} KB")
