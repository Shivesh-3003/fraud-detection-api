"""Average inference + processing time for the AE+MLP pipeline.

Replicates the production inference flow (preprocess → scale → AE → MLP) on
raw transactions from creditcard.csv, using the trained models. Reports per-
stage timing (mean, median, p95, p99) so we can compare against the Go API's
recorded `inference_time_ms` and `processing_time_ms` fields.
"""
from __future__ import annotations

import time
import json
import math
import warnings
import numpy as np
import pandas as pd
import joblib
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from _shared import DATA_DIR, MODELS_DIR, OUT_DIR, Autoencoder, FraudClassifier


def load_pipeline_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    ae = Autoencoder(input_dim=31)
    ae.load_state_dict(torch.load(MODELS_DIR / "autoencoder_model.pth", map_location="cpu"))
    ae.eval()
    clf = FraudClassifier(input_dim=32)
    clf.load_state_dict(torch.load(MODELS_DIR / "mlp_classifier.pth", map_location="cpu"))
    clf.eval()
    return scaler, ae, clf


def preprocess_ulb(v_features: list[float], amount: float, time_val: float) -> np.ndarray:
    """Mirror inference.py::_preprocess_ulb."""
    amount_log = math.log1p(amount)
    hour = math.floor(time_val / 3600) % 24
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    return np.array(v_features + [amount_log, hour_sin, hour_cos], dtype=np.float32)


def percentile_table(name: str, samples_ms: np.ndarray) -> dict:
    return {
        "stage": name,
        "n": int(len(samples_ms)),
        "mean_ms": float(np.mean(samples_ms)),
        "median_ms": float(np.median(samples_ms)),
        "std_ms": float(np.std(samples_ms)),
        "min_ms": float(np.min(samples_ms)),
        "p95_ms": float(np.percentile(samples_ms, 95)),
        "p99_ms": float(np.percentile(samples_ms, 99)),
        "max_ms": float(np.max(samples_ms)),
    }


def main(n_samples: int = 5000, warmup: int = 100) -> None:
    print(f"Loading pipeline artifacts...")
    scaler, ae, clf = load_pipeline_artifacts()

    print(f"Loading raw transactions from creditcard.csv...")
    df = pd.read_csv(DATA_DIR / "creditcard.csv")
    df = df.sample(n=n_samples + warmup, random_state=42).reset_index(drop=True)

    rows = []
    for _, r in df.iterrows():
        v_features = [float(r[f"V{i}"]) for i in range(1, 29)]
        rows.append((v_features, float(r["Amount"]), float(r["Time"])))

    print(f"Warming up ({warmup} iterations)...")
    for v, a, t in rows[:warmup]:
        feats = preprocess_ulb(v, a, t)
        scaled = scaler.transform(feats.reshape(1, -1)).flatten()
        with torch.no_grad():
            t_ae = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            recon = ae(t_ae)
            err = torch.mean((t_ae - recon) ** 2).item()
            clf_in = torch.tensor(np.append(scaled, err), dtype=torch.float32).unsqueeze(0)
            _ = torch.sigmoid(clf(clf_in)).item()

    print(f"Timing {n_samples} transactions...")
    t_pre = np.zeros(n_samples)
    t_scale = np.zeros(n_samples)
    t_ae = np.zeros(n_samples)
    t_clf = np.zeros(n_samples)
    t_inference = np.zeros(n_samples)   # AE+CLF only (matches inference.py)
    t_total = np.zeros(n_samples)       # full pipeline (preprocess→scale→AE→CLF)

    for i, (v, a, t) in enumerate(rows[warmup:warmup + n_samples]):
        s_total = time.perf_counter()

        s = time.perf_counter()
        feats = preprocess_ulb(v, a, t)
        t_pre[i] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        scaled = scaler.transform(feats.reshape(1, -1)).flatten()
        t_scale[i] = (time.perf_counter() - s) * 1000

        # AE + classifier matches what inference_time_ms reports in production.
        s_inf = time.perf_counter()

        s = time.perf_counter()
        with torch.no_grad():
            t_ae_in = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            recon = ae(t_ae_in)
            err = torch.mean((t_ae_in - recon) ** 2).item()
        t_ae[i] = (time.perf_counter() - s) * 1000

        s = time.perf_counter()
        with torch.no_grad():
            clf_in = torch.tensor(np.append(scaled, err), dtype=torch.float32).unsqueeze(0)
            _ = torch.sigmoid(clf(clf_in)).item()
        t_clf[i] = (time.perf_counter() - s) * 1000

        t_inference[i] = (time.perf_counter() - s_inf) * 1000
        t_total[i] = (time.perf_counter() - s_total) * 1000

    summary = [
        percentile_table("preprocess", t_pre),
        percentile_table("scale", t_scale),
        percentile_table("autoencoder_forward", t_ae),
        percentile_table("classifier_forward", t_clf),
        percentile_table("ml_inference (AE+CLF)", t_inference),
        percentile_table("end_to_end (full pipeline)", t_total),
    ]

    df_out = pd.DataFrame(summary)
    print("\n" + "=" * 100)
    print("AE+MLP INFERENCE TIMING (per single transaction)")
    print("=" * 100)
    print(df_out.to_string(index=False))
    print("=" * 100)

    avg_inf = float(np.mean(t_inference))
    avg_total = float(np.mean(t_total))
    print(f"\n>>> AVERAGE ML INFERENCE TIME:   {avg_inf:.4f} ms  ({1000/avg_inf:,.0f} txn/sec)")
    print(f">>> AVERAGE PROCESSING TIME:     {avg_total:.4f} ms  ({1000/avg_total:,.0f} txn/sec)")

    out_csv = OUT_DIR / "05_inference_timing.csv"
    out_json = OUT_DIR / "05_inference_timing.json"
    df_out.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "n_samples": n_samples,
            "warmup": warmup,
            "device": "cpu",
            "avg_ml_inference_ms": avg_inf,
            "avg_end_to_end_ms": avg_total,
            "stages": summary,
        }, f, indent=2)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_json}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(t_inference, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(avg_inf, color="red", ls="--",
                    label=f"mean = {avg_inf:.3f} ms")
    axes[0].axvline(np.percentile(t_inference, 95), color="orange", ls="--",
                    label=f"p95 = {np.percentile(t_inference, 95):.3f} ms")
    axes[0].set_xlabel("ML inference time (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"AE + MLP forward pass (n={n_samples})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(t_total, bins=60, color="seagreen", edgecolor="white")
    axes[1].axvline(avg_total, color="red", ls="--",
                    label=f"mean = {avg_total:.3f} ms")
    axes[1].axvline(np.percentile(t_total, 95), color="orange", ls="--",
                    label=f"p95 = {np.percentile(t_total, 95):.3f} ms")
    axes[1].set_xlabel("End-to-end processing time (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Full pipeline preprocess→scale→AE→MLP (n={n_samples})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Inference Timing Distribution")
    fig.tight_layout()
    out_plot = OUT_DIR / "05_inference_timing.png"
    fig.savefig(out_plot, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
