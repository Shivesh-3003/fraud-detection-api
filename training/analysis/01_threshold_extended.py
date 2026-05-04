"""F1 vs Threshold — extended sweep that does NOT stop at the optimum,
with finer granularity in the 0.9–1.0 range where the model concentrates
its high-confidence predictions.

Reuses the trained MLP + processed test set — no retraining.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

from _shared import load_test_set, predict_proba_mlp, OUT_DIR


def main() -> None:
    X_test, y_test = load_test_set()
    print(f"Test shape: {X_test.shape}, fraud: {int(y_test.sum())}/{len(y_test)}")

    y_probs = predict_proba_mlp(X_test)

    # Coarse grid 0.0–0.9 (step 0.02), fine grid 0.9–1.0 (step 0.002)
    coarse = np.arange(0.00, 0.90, 0.02)
    fine = np.arange(0.90, 1.000, 0.002)
    thresholds = np.unique(np.concatenate([coarse, fine, [0.5]]))

    f1s, precs, recs = [], [], []
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        precs.append(precision_score(y_test, y_pred, zero_division=0))
        recs.append(recall_score(y_test, y_pred, zero_division=0))
    f1s, precs, recs = map(np.array, (f1s, precs, recs))

    optimal_idx = int(np.argmax(f1s))
    optimal_t = float(thresholds[optimal_idx])
    optimal_f1 = float(f1s[optimal_idx])
    print(f"Optimal threshold: {optimal_t:.4f}  F1={optimal_f1:.4f}")

    # ---- Full-range plot (continues past optimum) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1s, color="steelblue", lw=2, label="F1-Score")
    ax.plot(thresholds, precs, color="seagreen", lw=1.3, ls="--", alpha=0.7, label="Precision")
    ax.plot(thresholds, recs, color="tomato", lw=1.3, ls="--", alpha=0.7, label="Recall")
    ax.axvline(optimal_t, color="red", ls=":", lw=1.5,
               label=f"Optimal = {optimal_t:.4f} (F1={optimal_f1:.3f})")
    ax.scatter([optimal_t], [optimal_f1], color="red", s=60, zorder=5)
    # Annotate the post-optimum region so the report can refer to it
    ax.axvspan(optimal_t, 1.0, color="red", alpha=0.05,
               label="Post-optimum region")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimisation — Full Range (Precision / Recall / F1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=2)
    fig.tight_layout()
    out1 = OUT_DIR / "01_threshold_full_range.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # ---- Zoomed plot focused on 0.9–1.0 (fine granularity) ----
    mask = thresholds >= 0.85
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds[mask], f1s[mask], color="steelblue", lw=2, marker="o",
            markersize=3, label="F1-Score")
    ax.plot(thresholds[mask], precs[mask], color="seagreen", lw=1.3, ls="--",
            alpha=0.8, label="Precision")
    ax.plot(thresholds[mask], recs[mask], color="tomato", lw=1.3, ls="--",
            alpha=0.8, label="Recall")
    ax.axvline(optimal_t, color="red", ls=":", lw=1.5,
               label=f"Optimal = {optimal_t:.4f}")
    ax.scatter([optimal_t], [optimal_f1], color="red", s=70, zorder=5)
    ax.set_xlim(0.85, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Optimisation — Fine Granularity (0.85–1.00)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out2 = OUT_DIR / "01_threshold_fine_zoom.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
