
# not fully cleaned up yet
"""
Generate visualisations of classifier training results.

"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from config import WORK_DIR, REGIME_LABELS, STANCE_LABELS

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Paths and figure-writing helpers

REPORT_DIR = os.path.join(WORK_DIR, "training_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

_written = []

def wrote(path):
    _written.append(path)
    print("Wrote:", path)

def savefig(path):
    """Lay out, save, and close the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    wrote(path)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Display settings

REGIME_DISPLAY_NAMES = {
    "Extraction_Dispossession":                "Extraction & Dispossession",
    "Human_Essence_Ontology":                  "Human Essence & Ontology",
    "Aesthetic_Pollution_Epistemic_Corruption": "Aesthetic Pollution",
    "Governance_Boundary_Policing":            "Governance & Boundary Policing",
    "Ideology_Hype_Discourse_Wars":            "Ideology, Hype & Discourse Wars",
    "AI_Native_Subculture_Legitimation":       "AI-Native Subculture",
    "Human_Artist_Community_Reproduction":     "Human Artist Community",
    "Adult_Content_NSFWAIGen":                 "Adult Content & NSFW AI",
    "Other_Unclear":                           "Other / Unclear",
}

LABEL_COLORS = {
    "Extraction_Dispossession":                "#e63946",
    "Human_Essence_Ontology":                  "#457b9d",
    "Aesthetic_Pollution_Epistemic_Corruption": "#f4a261",
    "Governance_Boundary_Policing":            "#2a9d8f",
    "Ideology_Hype_Discourse_Wars":            "#9b2226",
    "AI_Native_Subculture_Legitimation":       "#6a4c93",
    "Human_Artist_Community_Reproduction":     "#40916c",
    "Adult_Content_NSFWAIGen":                 "#e9c46a",
    "Other_Unclear":                           "#adb5bd",
    "pro":     "#2d6a4f",
    "anti":    "#c1121f",
    "mixed":   "#f4a261",
    "neutral": "#adb5bd",
}

def dn(label):
    return REGIME_DISPLAY_NAMES.get(label, label)

def lc(label, fallback="steelblue"):
    return LABEL_COLORS.get(label, fallback)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Load most recent metrics JSON

def load_metrics():
 # read and return the most recently modified training_metrics_*.json file  
    pattern = os.path.join(WORK_DIR, "training_metrics_*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No training_metrics_*.json found in {WORK_DIR}.\n"
            "Run 03_train_models.py first."
        )
    path = max(matches, key=os.path.getmtime)
    print(f"Loading metrics from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Class distribution

def plot_class_distribution(metrics):
    """Plot per-regime and per-stance training-example counts."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Regime distribution. Iterate REGIME_LABELS so the bar order is fixed across runs
    regime_dist  = metrics["regime_dist"]
    regimes      = [r for r in REGIME_LABELS if r in regime_dist]
    counts       = [regime_dist[r] for r in regimes]
    display_names = [dn(r) for r in regimes]
    colors        = [lc(r) for r in regimes]

    bars = axes[0].barh(display_names, counts, color=colors)
    axes[0].set_xlabel("Number of labeled examples")
    axes[0].set_title("Training examples per regime")
    axes[0].invert_yaxis()

    # numeric labels
    for bar, count in zip(bars, counts):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     str(count), va="center", fontsize=9)

    # stance distribution
    stance_dist   = metrics["stance_dist"]
    stances       = [s for s in STANCE_LABELS if s in stance_dist]
    stance_counts = [stance_dist[s] for s in stances]
    stance_colors = [lc(s) for s in stances]

    bars2 = axes[1].bar(stances, stance_counts, color=stance_colors)
    axes[1].set_ylabel("Number of labeled examples")
    axes[1].set_title("Training examples per stance")

    for bar, count in zip(bars2, stance_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha="center", fontsize=10)

    fig.suptitle(
        f"Training data class distribution  |  stamp: {metrics.get('stamp', 'unknown')}",
        fontsize=11
    )
    savefig(os.path.join(REPORT_DIR, "class_distribution.png"))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# stance confusion matrix

def plot_stance_confusion_matrix(metrics):
    """Plot the stance confusion matrix in raw counts and row-normalised form."""
    stance_data = metrics["stance"]
    classes     = stance_data["classes"]
    cm          = np.array(stance_data["confusion_matrix"])

    # row-normalise, np.where guard prevents a divide-by-zero
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: raw counts
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(len(classes)))
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_xticklabels(classes, rotation=30, ha="right")
    axes[0].set_yticklabels(classes)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion matrix — raw counts")
    plt.colorbar(im0, ax=axes[0])

    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[0].text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() * 0.6 else "black",
                         fontsize=10)

    # right: row-normalised proportions
    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(classes)))
    axes[1].set_yticks(range(len(classes)))
    axes[1].set_xticklabels(classes, rotation=30, ha="right")
    axes[1].set_yticklabels(classes)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion matrix — row-normalized proportions")
    plt.colorbar(im1, ax=axes[1])
    for i in range(len(classes)):
        for j in range(len(classes)):
            axes[1].text(j, i, f"{cm_norm[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if cm_norm[i, j] > 0.6 else "black",
                         fontsize=9)

    fig.suptitle("Stance classifier — confusion matrix", fontsize=12)
    savefig(os.path.join(REPORT_DIR, "stance_confusion_matrix.png"))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 3. Regime per-class metrics bar chart

def plot_regime_metrics(metrics):
 # plot per-regime precision, recall, F1 and support   

    regime_data = metrics["regime"]["per_class"]
    regimes     = [r for r in REGIME_LABELS if r in regime_data]

    precision = [regime_data[r]["precision"] for r in regimes]
    recall    = [regime_data[r]["recall"]    for r in regimes]
    f1        = [regime_data[r]["f1"]        for r in regimes]
    support   = [regime_data[r]["support"]   for r in regimes]
    display   = [dn(r) for r in regimes]

    x    = np.arange(len(regimes))
    w    = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 11),
                             gridspec_kw={"height_ratios": [3, 1]})

    # grouped bars: precision (left), recall (centre), F1 (right).
    axes[0].bar(x - w, precision, w, label="Precision", color="#457b9d", alpha=0.85)
    axes[0].bar(x,     recall,    w, label="Recall",    color="#e63946", alpha=0.85)
    axes[0].bar(x + w, f1,        w, label="F1",        color="#2a9d8f", alpha=0.85)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Regime classifier — per-class precision, recall, F1\n"
                       "(validation set, tuned thresholds)")
    axes[0].legend()
    # 0.7 reference line: an eyeballing aid, not a hard threshold.
    axes[0].axhline(0.7, color="gray", linestyle="--", linewidth=0.8,
                    label="0.7 reference line")

    # F1 value labels on top of the F1 bar group
    for i, v in enumerate(f1):
        axes[0].text(x[i] + w, v + 0.02, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=8)

    # support panel: per-regime colour
    axes[1].bar(x, support, color=[lc(r) for r in regimes], alpha=0.75)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Training examples")
    axes[1].set_title("Support (validation set examples per regime)")

    fig.suptitle(
        f"Regime classifier performance  |  stamp: {metrics.get('stamp', 'unknown')}",
        fontsize=11
    )
    savefig(os.path.join(REPORT_DIR, "regime_metrics.png"))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 4. Chosen threshold and F1 per regime

def plot_regime_thresholds(metrics):

    regime_data = metrics["regime"]["per_class"]
    regimes     = [r for r in REGIME_LABELS if r in regime_data]
    thresholds  = [regime_data[r]["threshold"] for r in regimes]
    f1_scores   = [regime_data[r]["f1"]        for r in regimes]
    display     = [dn(r) for r in regimes]

    x = np.arange(len(regimes))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w / 2, thresholds, w,
                    color="#6a4c93", alpha=0.8, label="Chosen threshold")
    bars2 = ax2.bar(x + w / 2, f1_scores,  w,
                    color="#2a9d8f", alpha=0.8, label="Validation F1")

    ax1.set_xticks(x)
    ax1.set_xticklabels(display, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Threshold", color="#6a4c93")
    ax1.set_ylim(0, 1.1)
    ax2.set_ylabel("F1 score", color="#2a9d8f")
    ax2.set_ylim(0, 1.1)

    # default 0.5 reference line on the threshold axis
    ax1.axhline(0.5, color="#6a4c93", linestyle="--", linewidth=0.7, alpha=0.5)

    # numeric value labels on each bar
    for bar, val in zip(bars1, thresholds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#6a4c93")
    for bar, val in zip(bars2, f1_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="#2a9d8f")

    # Combine the two axes' legends into a single box
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Regime classifier — chosen threshold and validation F1 per class\n"
              "(dashed line = default 0.5 threshold)")
    savefig(os.path.join(REPORT_DIR, "regime_thresholds.png"))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 5. Full threshold curves

def plot_threshold_curves(metrics):

    curves  = metrics["regime"].get("threshold_curves", {})
    regimes = metrics["regime"]["per_class"]

    if not curves:
        print("No threshold curve data found — skipping threshold_curves.png")
        print("(This is expected if running against an older metrics JSON.)")
        return

    n       = len(REGIME_LABELS)
    ncols   = 3
    nrows   = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), sharey=True)
    axes_flat = axes.flatten()

    for idx, regime in enumerate(REGIME_LABELS):
        ax     = axes_flat[idx]
        curve  = curves.get(regime, [])
        chosen = regimes.get(regime, {}).get("threshold", 0.5)
        color  = lc(regime)

        if curve:
            ts = [pt["threshold"] for pt in curve]
            fs = [pt["f1"]        for pt in curve]
            ax.plot(ts, fs, color=color, linewidth=2)
            # vertical dashed line: the chosen threshold for this regime.
            ax.axvline(chosen, color="black", linestyle="--",
                       linewidth=1, label=f"chosen={chosen:.2f}")
            # Horizontal dotted line: the peak F1 attained anywhere on the grid
            best_f1 = max(fs) if fs else 0
            ax.axhline(best_f1, color="gray", linestyle=":", linewidth=0.8)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        ax.set_title(dn(regime), fontsize=9)
        ax.set_xlabel("Threshold", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.set_xlim(0.15, 0.85)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    for idx in range(len(REGIME_LABELS), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Regime classifier — F1 vs. threshold curves (validation set)\n"
                 "Dashed line = chosen threshold", fontsize=11)
    savefig(os.path.join(REPORT_DIR, "threshold_curves.png"))

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main

def main():
 # loads the latest metrics JSCOn and writes all five training figures
    print("03b_visualize_training.py")
    print(f"Output directory: {REPORT_DIR}\n")

    metrics = load_metrics()

    print(f"Training stamp: {metrics.get('stamp', 'unknown')}")
    print(f"Stance training rows: {metrics.get('n_stance_train', '?'):,}")
    print(f"Regime training rows: {metrics.get('n_regime_train', '?'):,}\n")

    plot_class_distribution(metrics)
    plot_stance_confusion_matrix(metrics)
    plot_regime_metrics(metrics)
    plot_regime_thresholds(metrics)
    plot_threshold_curves(metrics)

    print(f"\nAll outputs written ({len(_written)} files):")
    for p in _written:
        print(f"  {p}")

if __name__ == "__main__":
    main()