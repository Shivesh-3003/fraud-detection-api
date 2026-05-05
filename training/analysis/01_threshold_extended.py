"""F1 / Precision / Recall vs threshold for the AE+MLP detector.

Produces three artifacts per dataset (--dataset {ulb,sparkov}):

    {dataset}_threshold_full_range.png   full 0-1 sweep
    {dataset}_threshold_fine_zoom.png    zoomed window near the optimum
    {dataset}_threshold_two_panel.png    both panels stacked (figure-ready)

The full-range sweep extends right up to threshold=1.0 (using every unique
predicted probability from sklearn's precision_recall_curve so the curves
are accurate, not just sampled at evenly spaced points). The operating
point dot uses the actual F1 at the optimal threshold, so it always lands
on the F1 curve rather than floating off it.
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, recall_score,
)

from _shared import load_test_set, predict_proba_mlp, load_optimal_threshold, OUT_DIR


def compute_curves(y_true: np.ndarray, y_probs: np.ndarray):
    """Return thresholds, precision, recall, F1 — one entry per unique threshold
    in y_probs, plus a synthetic threshold=1.0 endpoint so the curves close
    cleanly at the right edge."""
    p, r, t = precision_recall_curve(y_true, y_probs)
    # sklearn omits the threshold for the last (precision=1, recall=0) point.
    # Replicate by appending a threshold of 1.0 with precision=1, recall=0, f1=0.
    p = np.concatenate([p, [1.0]])
    r = np.concatenate([r, [0.0]])
    t = np.concatenate([t, [1.0]])
    # f1 paired with each (p, r); avoid div-by-zero
    f1 = 2 * p * r / (p + r + 1e-12)
    # precision/recall/f1 vectors are aligned with `t` after the append above
    # (the original sklearn arrays had len(t)+1; appending one to t balances).
    return t, p[:len(t)], r[:len(t)], f1[:len(t)]


def plot_full_range(ax, t, p, r, f1, opt_t, opt_f1):
    ax.plot(t, f1, color="steelblue", lw=2, label="F1-Score")
    ax.plot(t, p, color="seagreen", lw=1.3, ls="--", alpha=0.8, label="Precision")
    ax.plot(t, r, color="tomato", lw=1.3, ls="--", alpha=0.8, label="Recall")
    ax.axvline(opt_t, color="red", ls=":", lw=1.5,
               label=f"Optimal = {opt_t:.4f} (F1={opt_f1:.3f})")
    ax.scatter([opt_t], [opt_f1], color="red", s=60, zorder=5)
    ax.axvspan(opt_t, 1.0, color="red", alpha=0.05, label="Post-optimum region")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower center", ncol=2)


def plot_zoom(ax, t, p, r, f1, opt_t, opt_f1, lo: float, hi: float):
    mask = (t >= lo) & (t <= hi)
    ax.plot(t[mask], f1[mask], color="steelblue", lw=2, marker="o",
            markersize=3, label="F1-Score")
    ax.plot(t[mask], p[mask], color="seagreen", lw=1.3, ls="--", alpha=0.8,
            label="Precision")
    ax.plot(t[mask], r[mask], color="tomato", lw=1.3, ls="--", alpha=0.8,
            label="Recall")
    ax.axvline(opt_t, color="red", ls=":", lw=1.5,
               label=f"Optimal = {opt_t:.4f}")
    ax.scatter([opt_t], [opt_f1], color="red", s=70, zorder=5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ulb", "sparkov"], default="ulb")
    parser.add_argument("--zoom-lo", type=float, default=None,
                        help="Lower bound of zoom panel (default: 0.85 ULB / 0.95 sparkov)")
    parser.add_argument("--zoom-hi", type=float, default=1.00)
    args = parser.parse_args()

    if args.zoom_lo is None:
        args.zoom_lo = 0.85 if args.dataset == "ulb" else 0.95

    X_test, y_test = load_test_set(args.dataset)
    print(f"[{args.dataset}] Test shape: {X_test.shape}, "
          f"fraud: {int(y_test.sum())}/{len(y_test)}")

    y_probs = predict_proba_mlp(X_test, dataset=args.dataset)

    t, p, r, f1 = compute_curves(y_test, y_probs)

    # Use the saved threshold (from train_classifier.py) so the dot matches
    # what the production model actually uses; fall back to argmax if missing.
    try:
        opt_t = load_optimal_threshold(args.dataset)
    except FileNotFoundError:
        opt_t = float(t[int(np.argmax(f1))])

    # Pin opt_f1 to the actual F1 at this threshold (so the dot is on the curve).
    y_pred_opt = (y_probs >= opt_t).astype(int)
    opt_f1 = f1_score(y_test, y_pred_opt, zero_division=0)
    opt_p = precision_score(y_test, y_pred_opt, zero_division=0)
    opt_r = recall_score(y_test, y_pred_opt, zero_division=0)
    print(f"[{args.dataset}] Optimal threshold: {opt_t:.6f}  "
          f"F1={opt_f1:.4f}  P={opt_p:.4f}  R={opt_r:.4f}")

    title_ds = args.dataset.upper()

    # 1. Full-range panel
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_full_range(ax, t, p, r, f1, opt_t, opt_f1)
    ax.set_title(f"Threshold Optimisation — Full Range ({title_ds})")
    fig.tight_layout()
    out1 = OUT_DIR / f"{args.dataset}_threshold_full_range.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)
    print(f"Saved: {out1}")

    # 2. Zoom panel
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_zoom(ax, t, p, r, f1, opt_t, opt_f1, args.zoom_lo, args.zoom_hi)
    ax.set_title(
        f"Threshold Optimisation — Fine Granularity "
        f"({args.zoom_lo:.2f}–{args.zoom_hi:.2f}, {title_ds})"
    )
    fig.tight_layout()
    out2 = OUT_DIR / f"{args.dataset}_threshold_fine_zoom.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"Saved: {out2}")

    # 3. Combined two-panel figure for the report
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 11))
    plot_full_range(ax_top, t, p, r, f1, opt_t, opt_f1)
    ax_top.set_title(f"Top: Full Range (0.0–1.0)")
    plot_zoom(ax_bot, t, p, r, f1, opt_t, opt_f1, args.zoom_lo, args.zoom_hi)
    ax_bot.set_title(
        f"Bottom: Fine Granularity ({args.zoom_lo:.2f}–{args.zoom_hi:.2f})"
    )
    fig.suptitle(f"Threshold Optimisation — {title_ds} Dataset", y=0.995)
    fig.tight_layout()
    out3 = OUT_DIR / f"{args.dataset}_threshold_two_panel.png"
    fig.savefig(out3, dpi=150)
    plt.close(fig)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
