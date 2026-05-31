#!/usr/bin/env python3
"""Generate hypersphere concept visualizations for HYPERSPHERE.md."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#1a1a2e"
TEXT    = "#e0e0e0"
GREEN   = "#76b900"
BLUE    = "#4477ff"
ORANGE  = "#cc6622"
CYAN    = "#44cccc"
YELLOW  = "#ffcc00"
PURPLE  = "#aa44ff"
RED     = "#cc2244"
GREY    = "#555577"

COLORS6 = [GREEN, BLUE, ORANGE, CYAN, YELLOW, PURPLE]

def savefig(fig, name):
    for ext in ("png", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
    print(f"  saved {name}.png/.svg")
    plt.close(fig)

def base_fig(w=10, h=6, nrows=1, ncols=1, **kw):
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h),
                           facecolor=BG, **kw)
    return fig, ax

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GREY)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    if title:  ax.set_title(title, color=TEXT, fontsize=12, pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT)

def style_ax3d(ax, title=""):
    ax.set_facecolor(BG)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(GREY)
    ax.yaxis.pane.set_edgecolor(GREY)
    ax.zaxis.pane.set_edgecolor(GREY)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.zaxis.label.set_color(TEXT)
    if title: ax.set_title(title, color=TEXT, fontsize=11, pad=4)


# ── hyper1: L2 normalization on unit circle ───────────────────────────────────
print("Rendering hyper1_normalization ...")

raw_dirs = np.array([[0.8, 0.6], [0.3, 0.95], [-0.5, 0.87],
                     [-0.95, 0.31], [-0.7, -0.71], [0.6, -0.8]])
raw_mags = np.array([2.1, 1.4, 1.8, 0.7, 2.5, 1.1])
raw_vecs = raw_dirs * raw_mags[:, None]
norm_vecs = raw_dirs  # already unit

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=BG)
fig.suptitle("L2 Normalization: Direction Preserved, Magnitude Discarded",
             color=TEXT, fontsize=14, y=1.01)

for col, (vecs, title, show_circle) in enumerate([
    (raw_vecs,  "Raw embedding vectors\n(different magnitudes)", False),
    (norm_vecs, "After L2 normalization\n(all on unit circle)", True),
]):
    ax = axes[col]
    style_ax(ax, title=title)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.8, zorder=0)
    ax.axvline(0, color=GREY, lw=0.8, zorder=0)

    if show_circle:
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color=GREEN, lw=1.5,
                linestyle="--", alpha=0.7, label="unit circle")
        ax.text(1.05, 0.05, "r=1", color=GREEN, fontsize=9)

    for i, (v, c) in enumerate(zip(vecs, COLORS6)):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.2))
        ax.scatter(*v, color=c, s=60, zorder=5)
        ax.text(v[0]*1.08, v[1]*1.08, f"d{i+1}", color=c, fontsize=9,
                ha="center", va="center")

    if not show_circle:
        # dashed magnitude bars
        for v, c in zip(vecs, COLORS6):
            mag = np.linalg.norm(v)
            ax.text(v[0]/2, v[1]/2 - 0.15, f"|v|={mag:.1f}",
                    color=c, fontsize=7.5, alpha=0.8)

ax.legend(loc="lower right", facecolor=BG, labelcolor=TEXT, fontsize=9)
axes[0].text(0, -2.55,
    "Docs of different lengths land at\ndifferent distances from origin",
    color=TEXT, fontsize=9, ha="center", alpha=0.8)
axes[1].text(0, -2.55,
    "All tips sit on the unit circle.\nAngle between vectors = semantic distance.",
    color=GREEN, fontsize=9, ha="center")

plt.tight_layout()
savefig(fig, "hyper1_normalization")


# ── hyper2: 3D sphere with class patches ──────────────────────────────────────
print("Rendering hyper2_sphere_patches ...")

def sphere():
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    return (np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)))

def cluster_on_sphere(center_theta, center_phi, n=40, spread=0.18):
    rng = np.random.default_rng(int(center_theta*100 + center_phi*100))
    thetas = center_theta + rng.normal(0, spread, n)
    phis   = center_phi   + rng.normal(0, spread, n)
    x = np.sin(phis) * np.cos(thetas)
    y = np.sin(phis) * np.sin(thetas)
    z = np.cos(phis)
    norms = np.sqrt(x**2 + y**2 + z**2)
    return x/norms, y/norms, z/norms

CLASSES = [
    ("strategy",  0.5,  1.1, GREEN),
    ("medical",   1.8,  0.9, CYAN),
    ("weapons",   3.5,  1.0, ORANGE),
    ("C2",        2.5,  2.0, BLUE),
    ("Psyop",     0.3,  2.3, YELLOW),
    ("mines",     4.5,  1.5, PURPLE),
]

fig = plt.figure(figsize=(10, 8), facecolor=BG)
ax  = fig.add_subplot(111, projection="3d")
style_ax3d(ax, title="Documents as points on the unit hypersphere  (shown in 3D, actual: 1024D)")
ax.set_facecolor(BG)

# wireframe sphere
X, Y, Z = sphere()
ax.plot_wireframe(X, Y, Z, color=GREY, alpha=0.12, linewidth=0.4, rcount=20, ccount=20)

# class clusters
for cls, ct, cp, col in CLASSES:
    cx, cy, cz = cluster_on_sphere(ct, cp, n=35, spread=0.15)
    ax.scatter(cx, cy, cz, color=col, s=22, alpha=0.85, depthshade=True)
    # label at centroid
    lx, ly, lz = cx.mean()*1.25, cy.mean()*1.25, cz.mean()*1.25
    ax.text(lx, ly, lz, cls, color=col, fontsize=9.5, fontweight="bold", ha="center")

ax.set_xlabel("dim 1", color=TEXT, labelpad=-4)
ax.set_ylabel("dim 2", color=TEXT, labelpad=-4)
ax.set_zlabel("dim 3", color=TEXT, labelpad=-4)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.4, 1.4)
ax.view_init(elev=22, azim=38)

fig.text(0.5, 0.02,
    "In reality: 1024 dimensions — impossible to visualise, but the geometry is identical.",
    ha="center", color=TEXT, fontsize=9, alpha=0.7)

plt.tight_layout()
savefig(fig, "hyper2_sphere_patches")


# ── hyper3: cosine similarity ─────────────────────────────────────────────────
print("Rendering hyper3_cosine_similarity ...")

def draw_cos_panel(ax, angle_deg, label, col_a, col_b):
    theta = np.radians(angle_deg)
    u = np.array([1.0, 0.0])
    v = np.array([np.cos(theta), np.sin(theta)])
    cos_val = np.dot(u, v)

    style_ax(ax)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)

    # unit circle faint
    th = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=GREY, lw=0.8, alpha=0.5)

    # vectors
    ax.annotate("", xy=u, xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=col_a, lw=2.5))
    ax.annotate("", xy=v, xytext=(0,0),
                arrowprops=dict(arrowstyle="->", color=col_b, lw=2.5))
    ax.text(u[0]+0.05, u[1]+0.05, "u", color=col_a, fontsize=11, fontweight="bold")
    ax.text(v[0]+0.05, v[1]+0.05, "v", color=col_b, fontsize=11, fontweight="bold")

    # arc
    arc_t = np.linspace(0, theta, 60)
    r = 0.32
    ax.plot(r*np.cos(arc_t), r*np.sin(arc_t), color=YELLOW, lw=1.5)
    mid = theta/2
    ax.text(r*1.4*np.cos(mid), r*1.4*np.sin(mid), "θ",
            color=YELLOW, fontsize=11, ha="center")

    # cos value
    sign = "+" if cos_val >= 0 else ""
    ax.set_title(f"{label}\ncos θ = {sign}{cos_val:.2f}",
                 color=TEXT, fontsize=11, pad=6)

    # semantic annotation
    if cos_val > 0.7:
        sem = "very similar"
        sc = GREEN
    elif cos_val > 0.2:
        sem = "somewhat related"
        sc = CYAN
    elif cos_val > -0.2:
        sem = "unrelated"
        sc = TEXT
    else:
        sem = "opposite meaning"
        sc = RED
    ax.text(0, -1.3, sem, color=sc, fontsize=9, ha="center", style="italic")

fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=BG)
fig.suptitle("Cosine Similarity — Angle Between Embedding Vectors = Semantic Distance",
             color=TEXT, fontsize=13, y=1.02)

draw_cos_panel(axes[0], 18,  "Nearly the same direction", GREEN, CYAN)
draw_cos_panel(axes[1], 90,  "Perpendicular", ORANGE, BLUE)
draw_cos_panel(axes[2], 162, "Opposite directions", RED, YELLOW)

fig.text(0.5, -0.04,
    "Formula:  cos(θ) = u · v  (dot product of unit vectors)\n"
    "bge-m3 encodes documents into unit vectors — angle between them = how different they are.",
    ha="center", color=TEXT, fontsize=9.5, alpha=0.85)

plt.tight_layout()
savefig(fig, "hyper3_cosine_similarity")


# ── hyper4: decision boundary ─────────────────────────────────────────────────
print("Rendering hyper4_decision_boundary ...")

fig = plt.figure(figsize=(10, 8), facecolor=BG)
ax  = fig.add_subplot(111, projection="3d")
style_ax3d(ax, title="Decision Boundary: Hyperplane Slices the Sphere into Class Regions")

X, Y, Z = sphere()
ax.plot_wireframe(X, Y, Z, color=GREY, alpha=0.10, linewidth=0.4, rcount=18, ccount=18)

# two hemisphere point clouds
rng = np.random.default_rng(7)
N   = 200
phi   = rng.uniform(0, np.pi, N)
theta = rng.uniform(0, 2*np.pi, N)
px = np.sin(phi)*np.cos(theta)
py = np.sin(phi)*np.sin(theta)
pz = np.cos(phi)

# decision: px + 0.3*py > 0  → class A, else class B
mask = (px + 0.3*py) > 0
col_A = np.array([118/255, 185/255, 0/255, 0.6])   # green
col_B = np.array([204/255, 102/255, 34/255, 0.6])   # orange

ax.scatter(px[mask],  py[mask],  pz[mask],  color=col_A, s=14, depthshade=True)
ax.scatter(px[~mask], py[~mask], pz[~mask], color=col_B, s=14, depthshade=True)

# great circle (decision boundary in 3D)
t = np.linspace(0, 2*np.pi, 200)
normal = np.array([1.0, 0.3, 0.0])
normal /= np.linalg.norm(normal)
perp1 = np.cross(normal, [0,0,1]); perp1 /= np.linalg.norm(perp1)
perp2 = np.cross(normal, perp1)
gc = np.outer(np.cos(t), perp1) + np.outer(np.sin(t), perp2)
ax.plot(gc[:,0], gc[:,1], gc[:,2], color=GREEN, lw=2.5, zorder=10)

# normal vector w
ax.quiver(0, 0, 0, *normal*1.3, color=YELLOW, lw=2.5, arrow_length_ratio=0.12)
ax.text(*(normal*1.45), "w\n(class weight\nvector)", color=YELLOW,
        fontsize=9, ha="center")

# class labels
ax.text( 0.9,  0.4,  0.3, "class A", color=GREEN,  fontsize=11, fontweight="bold")
ax.text(-0.8, -0.2, -0.4, "class B", color=ORANGE, fontsize=11, fontweight="bold")
ax.text( 0,    0,  -1.5,
    "Green circle = decision boundary (great circle)\n"
    "LR score: w · doc_vec  →  positive = class A, negative = class B",
    color=TEXT, fontsize=8, ha="center")

ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_zlim(-1.8, 1.4)
ax.set_xlabel("dim 1", color=TEXT, labelpad=-4)
ax.set_ylabel("dim 2", color=TEXT, labelpad=-4)
ax.set_zlabel("dim 3", color=TEXT, labelpad=-4)
ax.view_init(elev=18, azim=55)
plt.tight_layout()
savefig(fig, "hyper4_decision_boundary")


# ── hyper5: full pipeline (Pillow) ────────────────────────────────────────────
print("Rendering hyper5_pipeline ...")

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 280
img  = Image.new("RGB", (W, H), (26, 26, 46))
draw = ImageDraw.Draw(img)

try:
    fnt_lg = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 15)
    fnt_sm = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 11)
    fnt_xs = ImageFont.truetype("/usr/share/fonts/truetype/liberation2/LiberationMono-Bold.ttf", 9)
except Exception:
    fnt_lg = fnt_sm = fnt_xs = ImageFont.load_default()

# grid lines
for y in range(0, H, 28):
    draw.line([(0, y), (W, y)], fill=(30, 30, 52), width=1)

STEPS = [
    ("Raw file\n.pdf/.jpg/.txt",    (26, 26, 46),   (100, 100, 160)),
    ("chunk_text()\n4000 chars",    (20, 35, 20),   (80, 160, 80)),
    ("bge-m3\nencode(chunks)",      (20, 20, 70),   (80, 80, 220)),
    ("mean-pool\nchunks → 1 vec",   (20, 35, 20),   (80, 160, 80)),
    ("L2 normalize\n‖v‖ = 1.0",     (40, 10, 10),   (200, 80, 80)),
    ("LogisticReg.\ndot product",   (40, 30, 0),    (200, 150, 0)),
    ("softmax\n→ class %",          (10, 40, 10),   (60, 185, 0)),
]

box_w, box_h = 108, 64
gap = 16
total_w = len(STEPS) * box_w + (len(STEPS)-1) * gap
x0 = (W - total_w) // 2
y0 = (H - box_h) // 2

for i, (label, bg, border) in enumerate(STEPS):
    bx = x0 + i * (box_w + gap)
    draw.rounded_rectangle([bx, y0, bx+box_w, y0+box_h], radius=6,
                            fill=bg, outline=border, width=2)
    lines = label.split("\n")
    for j, line in enumerate(lines):
        tw = draw.textlength(line, font=fnt_sm)
        tx = bx + (box_w - tw) // 2
        ty = y0 + 10 + j * 22
        draw.text((tx, ty), line, font=fnt_sm, fill=(220, 220, 220))

    if i < len(STEPS) - 1:
        ax_start = bx + box_w + 3
        ax_end   = bx + box_w + gap - 3
        ay = y0 + box_h // 2
        draw.line([(ax_start, ay), (ax_end, ay)], fill=(170, 170, 200), width=2)
        draw.polygon([(ax_end, ay-5), (ax_end+7, ay), (ax_end, ay+5)],
                     fill=(170, 170, 200))

# title + footer
title = "doc_classifier_gpu — Embedding Pipeline"
tw = draw.textlength(title, font=fnt_lg)
draw.text(((W - tw) // 2, 18), title, font=fnt_lg, fill=(220, 220, 220))

footer = "Each document → one 1024-dim unit vector → one point on the hypersphere → class prediction"
fw = draw.textlength(footer, font=fnt_xs)
draw.text(((W - fw) // 2, H - 26), footer, font=fnt_xs, fill=(130, 130, 160))

for ext, fmt in [("png", "PNG"), ("svg", "PNG")]:
    out_path = OUT / f"hyper5_pipeline.{ext}"
    img.save(out_path, fmt if fmt == "PNG" else "PNG")
print("  saved hyper5_pipeline.png/.svg")


print("\nAll done.")
