"""Generate index_alignment.png — run from the docs/_static/ directory."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ────────────────────────────────────────────────────────────────────
ALL_IDS  = [0, 1, 2, 3, 4]
A_VALS   = {0: 8, 1: 4, 3: 3, 4: 7}
B_VALS   = {0: 2, 1: 2, 2: 2, 4: 2}
COMMON   = {0: 6, 1: 2, 4: 5}          # c = a - b at common pulse IDs

# ── Layout ──────────────────────────────────────────────────────────────────
COL_X    = {pid: 1.6 + pid * 1.15 for pid in ALL_IDS}   # x centre per pulse ID
Y_A, Y_B, Y_C = 3.2, 1.9, 0.5                           # row y-centres
BW, BH   = 0.80, 0.52                                    # box width / height

C_A      = "#4878CF"   # blue
C_B      = "#E07B39"   # orange
C_C      = "#50A050"   # green
C_FADE   = "#D5D5D5"   # greyed-out (event excluded from operation)
C_LINE   = "#888888"   # connector lines

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.set_xlim(0, 8.0)
ax.set_ylim(-0.1, 4.2)
ax.axis("off")
fig.patch.set_facecolor("white")


def draw_box(cx, cy, val, color, alpha=1.0, fontsize=13):
    rect = mpatches.FancyBboxPatch(
        (cx - BW / 2, cy - BH / 2), BW, BH,
        boxstyle="round,pad=0.06",
        linewidth=1.4,
        edgecolor="white" if alpha < 0.5 else "#444444",
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, str(val), ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="white" if alpha > 0.45 else "#999999",
            zorder=4)


# ── Row labels ──────────────────────────────────────────────────────────────
label_kw = dict(ha="right", va="center", fontsize=12, fontweight="bold")
ax.text(1.05, Y_A, "a", color=C_A,   **label_kw)
ax.text(1.05, Y_B, "b", color=C_B,   **label_kw)
ax.text(1.05, Y_C, "c = a − b", color=C_C, **label_kw)

# ── Pulse-ID axis labels ─────────────────────────────────────────────────────
for pid in ALL_IDS:
    ax.text(COL_X[pid], 4.0, f"ID {pid}", ha="center", va="center",
            fontsize=9, color="#555555")

# ── Array a boxes ────────────────────────────────────────────────────────────
for pid in ALL_IDS:
    if pid in A_VALS:
        excluded = pid not in COMMON
        draw_box(COL_X[pid], Y_A, A_VALS[pid],
                 color=C_FADE if excluded else C_A,
                 alpha=0.35 if excluded else 1.0)

# ── Array b boxes ────────────────────────────────────────────────────────────
for pid in ALL_IDS:
    if pid in B_VALS:
        excluded = pid not in COMMON
        draw_box(COL_X[pid], Y_B, B_VALS[pid],
                 color=C_FADE if excluded else C_B,
                 alpha=0.35 if excluded else 1.0)

# ── Connector lines + minus signs between a and b ────────────────────────────
for pid in COMMON:
    x = COL_X[pid]
    ax.plot([x, x], [Y_A - BH / 2 - 0.05, Y_B + BH / 2 + 0.05],
            color=C_LINE, lw=1.2, ls="--", zorder=2)
    # minus sign between rows
    ax.text(x, (Y_A + Y_B) / 2, "−", ha="center", va="center",
            fontsize=14, color="#555555", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1))

# ── Connector lines between b and result ─────────────────────────────────────
for pid in COMMON:
    x = COL_X[pid]
    ax.plot([x, x], [Y_B - BH / 2 - 0.05, Y_C + BH / 2 + 0.05],
            color=C_LINE, lw=1.2, ls="--", zorder=2)
    ax.text(x, (Y_B + Y_C) / 2, "=", ha="center", va="center",
            fontsize=14, color="#555555", zorder=5,
            bbox=dict(fc="white", ec="none", pad=1))

# ── Result c boxes ───────────────────────────────────────────────────────────
for pid in COMMON:
    draw_box(COL_X[pid], Y_C, COMMON[pid], color=C_C)

# ── Annotation: excluded events ──────────────────────────────────────────────
# "only in a": text above-left of ID 3 faded box, arrow to its top
ax.annotate(
    "only in a\n→ excluded",
    xy=(COL_X[3], Y_A + BH / 2 + 0.05),
    xytext=(COL_X[3] - 1.4, Y_A + 0.50),
    fontsize=8, color="#888888",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                    connectionstyle="arc3,rad=-0.25"),
    ha="center", va="bottom",
)
# "only in b": text below-right of ID 2 faded box, arrow to its bottom
ax.annotate(
    "only in b\n→ excluded",
    xy=(COL_X[2], Y_B - BH / 2 - 0.05),
    xytext=(COL_X[2] + 1.2, Y_B - 0.72),
    fontsize=8, color="#888888",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.9,
                    connectionstyle="arc3,rad=0.25"),
    ha="center", va="top",
)

ax.set_title(
    "Index alignment: a − b keeps only common pulse IDs",
    fontsize=12, pad=8, color="#333333",
)

plt.tight_layout()
plt.savefig("index_alignment.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Saved index_alignment.png")
