"""ROC curve for the AE+MLP fraud detector. Reuses the trained MLP."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from _shared import load_test_set, predict_proba_mlp, load_optimal_threshold, OUT_DIR


def main() -> None:
    X_test, y_test = load_test_set()
    y_probs = predict_proba_mlp(X_test)

    fpr, tpr, thr = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    print(f"AUC-ROC = {auc:.4f}")

    op_thr = load_optimal_threshold("ulb")
    idx = int(np.argmin(np.abs(thr - op_thr)))
    op_fpr, op_tpr = fpr[idx], tpr[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AE+MLP (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="Random (AUC = 0.5)")
    ax.scatter([op_fpr], [op_tpr], color="red", s=80, zorder=5,
               label=f"Operating point (thr={op_thr:.4f})\nTPR={op_tpr:.3f}, FPR={op_fpr:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve — AE+MLP Fraud Detector (ULB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "03_roc_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xlim, ylim, title in (
        (ax1, (0, 1), (0, 1.02), "Full ROC"),
        (ax2, (0, 0.05), (0.5, 1.02), "Zoom: low-FPR region"),
    ):
        ax.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
        ax.scatter([op_fpr], [op_tpr], color="red", s=70, zorder=5,
                   label=f"Op. pt (thr={op_thr:.4f})")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle("ROC Curve — AE+MLP Fraud Detector (ULB)")
    fig.tight_layout()
    out2 = OUT_DIR / "03_roc_curve_zoom.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
