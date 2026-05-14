"""Generate the PathwayML-Ath analysis workflow diagram (Fig. 1).

Produces a six-box flowchart showing the pipeline stages:
data sources -> feature engineering -> classification -> validation -> outputs.
Saved as figures/fig1_workflow_v6.{png,pdf} and figures/fig1_workflow.png.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

Path('figures').mkdir(exist_ok=True)

# ── Canvas setup ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.2))
ax.axis('off')

# ── Workflow boxes ─────────────────────────────────────────────────────
# Layout: two data-source boxes (left), feature + model (middle/right),
#          validation + outputs (bottom row).
# Each tuple: (x, y, width, height, label text) in axes-fraction coords.
boxes = [
    (0.05, 0.68, 0.22, 0.18, 'Curated pathways\nKEGG + AraCyc\n539 positives'),
    (0.05, 0.30, 0.22, 0.18, 'Synthetic decoys\nsize-matched, shuffled\ncorrupted, cross-pathway'),
    (0.36, 0.50, 0.24, 0.18, 'Training-only features\nGO frequency + Jaccard\n+ size'),
    (0.69, 0.50, 0.22, 0.18, 'XGBoost classifier\nno dense embedding\nseed + multiseed runs'),
    (0.36, 0.13, 0.24, 0.18, 'Validation analyses\nablation, negative type\nLOFO, candidates'),
    (0.69, 0.13, 0.22, 0.18, 'Outputs\ncoherence score\nuncertainty + interpretation')]

for x, y, w, h, text in boxes:
    # Rounded rectangle with subtle padding for a clean publication look
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.018,rounding_size=0.015',
        linewidth=1.2, edgecolor='black', facecolor='white')
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=10)

# ── Directed arrows between stages ────────────────────────────────────
# Arrows encode the data flow: sources -> features -> model -> outputs,
# plus a feedback path from model to validation.
arrows = [
    ((0.27, 0.77), (0.36, 0.60)),   # positives -> features
    ((0.27, 0.39), (0.36, 0.57)),   # negatives -> features
    ((0.60, 0.59), (0.69, 0.59)),   # features -> classifier
    ((0.80, 0.50), (0.80, 0.31)),   # classifier -> outputs
    ((0.60, 0.22), (0.69, 0.22)),   # validation -> outputs
    ((0.48, 0.50), (0.48, 0.31)),   # features -> validation
]
for a, b in arrows:
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle='-|>', mutation_scale=14, linewidth=1.1, color='black'))

# ── Title and export ───────────────────────────────────────────────────
ax.text(0.5, 0.94, 'PathwayML-Ath analysis workflow',
        ha='center', va='center', fontsize=14, weight='bold')
plt.tight_layout()
# Save both raster and vector formats; version-neutral alias for other scripts
plt.savefig('figures/fig1_workflow_v6.png', dpi=300)
plt.savefig('figures/fig1_workflow_v6.pdf')
plt.savefig('figures/fig1_workflow.png', dpi=300)
