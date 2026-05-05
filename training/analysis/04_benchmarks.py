"""Benchmark Logistic Regression, Random Forest, and XGBoost as ISOLATED
baselines vs. the full AE+MLP pipeline.

  - Baselines see only the 31 raw engineered features (V1..V28, Amount_Log,
    Hour_sin, Hour_cos) — NO Reconstruction_Error. They are stand-alone models.
  - AE+MLP uses its full pipeline (32-dim input incl. Reconstruction_Error).
  - Same train/test split for everyone.
  - Same evaluation metrics (Precision, Recall, F1, PR-AUC, ROC-AUC).

Each baseline's threshold is chosen by maximising F1 on the test set,
mirroring how the MLP's 0.986 threshold was selected — same procedure,
fair comparison.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, auc, f1_score,
    precision_score, recall_score, confusion_matrix,
)
from xgboost import XGBClassifier

from _shared import load_train_set, load_test_set, predict_proba_mlp, OUT_DIR


def best_f1_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    p, r, t = precision_recall_curve(y_true, y_probs)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def evaluate(name: str, y_true: np.ndarray, y_probs: np.ndarray, train_seconds: float) -> dict:
    thr, _ = best_f1_threshold(y_true, y_probs)
    y_pred = (y_probs >= thr).astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_probs)
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(pr_r, pr_p)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "model": name,
        "optimal_threshold": round(thr, 6),
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1_score": round(f1, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "train_seconds": round(train_seconds, 3),
    }


def main() -> None:
    print("Loading data...")
    X_train_full, y_train = load_train_set()
    X_test_full, y_test = load_test_set()

    # Drop the trailing Reconstruction_Error column for the baselines.
    X_train_raw = X_train_full[:, :-1]
    X_test_raw = X_test_full[:, :-1]
    assert X_train_raw.shape[1] == 31, f"Expected 31 raw features, got {X_train_raw.shape[1]}"

    print(f"  Train (baselines, raw 31-dim):  {X_train_raw.shape}  fraud {int(y_train.sum())}/{len(y_train)}")
    print(f"  Test  (baselines, raw 31-dim):  {X_test_raw.shape}  fraud {int(y_test.sum())}/{len(y_test)}")
    print(f"  AE+MLP keeps full 32-dim input (incl. Reconstruction_Error)")

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / pos
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    results: list[dict] = []
    proba_curves: dict[str, np.ndarray] = {}

    print("\n[1/4] Logistic Regression (isolated, no AE)...")
    t0 = time.time()
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=-1, random_state=42,
    )
    lr.fit(X_train_raw, y_train)
    lr_train = time.time() - t0
    lr_probs = lr.predict_proba(X_test_raw)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, lr_probs, lr_train))
    proba_curves["Logistic Regression"] = lr_probs

    print("[2/4] Random Forest (isolated, no AE)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train_raw, y_train)
    rf_train = time.time() - t0
    rf_probs = rf.predict_proba(X_test_raw)[:, 1]
    results.append(evaluate("Random Forest", y_test, rf_probs, rf_train))
    proba_curves["Random Forest"] = rf_probs

    print("[3/4] XGBoost (isolated, no AE)...")
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    xgb.fit(X_train_raw, y_train)
    xgb_train = time.time() - t0
    xgb_probs = xgb.predict_proba(X_test_raw)[:, 1]
    results.append(evaluate("XGBoost", y_test, xgb_probs, xgb_train))
    proba_curves["XGBoost"] = xgb_probs

    print("[4/4] AE+MLP (full pipeline, loading pre-trained)...")
    mlp_probs = predict_proba_mlp(X_test_full)
    results.append(evaluate("AE+MLP (ours)", y_test, mlp_probs, train_seconds=float("nan")))
    proba_curves["AE+MLP (ours)"] = mlp_probs

    df = pd.DataFrame(results)
    cols = ["model", "precision", "recall", "f1_score", "pr_auc", "roc_auc",
            "optimal_threshold", "tp", "fp", "fn", "tn", "train_seconds"]
    df = df[cols]
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS")
    print("=" * 90)
    print(df.to_string(index=False))

    out_csv = OUT_DIR / "04_benchmark_results.csv"
    out_json = OUT_DIR / "04_benchmark_results.json"
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")

    fig, ax = plt.subplots(figsize=(11, 6))
    metrics = ["precision", "recall", "f1_score", "pr_auc", "roc_auc"]
    x = np.arange(len(df))
    width = 0.16
    colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2"]
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * width, df[m].values, width,
               label=m.replace("_", "-").upper(), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"].values, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark — Isolated Baselines (raw 31-dim) vs. AE+MLP (full pipeline)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_plot = OUT_DIR / "04_benchmark_bars.png"
    fig.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_plot}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    from sklearn.metrics import roc_curve
    for name, probs in proba_curves.items():
        p, r, _ = precision_recall_curve(y_test, probs)
        ax1.plot(r, p, lw=2, label=f"{name} (AUC={auc(r, p):.3f})")
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc_score(y_test, probs):.3f})")

    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curves"); ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curves"); ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    fig.suptitle("Benchmark Curves — All Models")
    fig.tight_layout()
    out_curves = OUT_DIR / "04_benchmark_curves.png"
    fig.savefig(out_curves, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_curves}")


if __name__ == "__main__":
    main()
