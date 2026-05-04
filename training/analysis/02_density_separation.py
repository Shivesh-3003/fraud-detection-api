"""Reconstruction-error density plot.

Reuses X_train_final.csv (already contains Reconstruction_Error) joined to
y_train_MLP.csv. Renders a smooth normal (green, ~284k samples) curve and a
deliberately-jagged fraud (red, ~492 samples) curve to honour the actual
sample-size asymmetry. X-axis is extended to 22 to cover the full fraud tail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from _shared import DATA_DIR, OUT_DIR


def main() -> None:
    # Combine train + test so we use ALL ~284k normals and all ~492 frauds
    X_train = pd.read_csv(DATA_DIR / "X_train_final.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train_MLP.csv").values.flatten()
    X_test = pd.read_csv(DATA_DIR / "X_test_final.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.flatten()

    err = np.concatenate([
        X_train["Reconstruction_Error"].values,
        X_test["Reconstruction_Error"].values,
    ])
    y = np.concatenate([y_train, y_test])

    normal = err[y == 0]
    fraud = err[y == 1]
    print(f"Normal: n={len(normal):,}  range=[{normal.min():.4f}, {normal.max():.2f}]")
    print(f"Fraud:  n={len(fraud):,}    range=[{fraud.min():.4f}, {fraud.max():.2f}]")

    # X-axis grid
    x_grid = np.linspace(0, 22, 1000)

    # Normal: smooth Gaussian KDE (Scott's bandwidth → very smooth at n=284k)
    kde_normal = gaussian_kde(normal, bw_method="scott")
    pdf_normal = kde_normal(x_grid)

    # Fraud: deliberately under-smoothed using Silverman scaled down so the
    # 492-sample sparsity shows up as the natural jaggedness it actually has.
    kde_fraud = gaussian_kde(fraud, bw_method=lambda k: 0.15 * k.silverman_factor())
    pdf_fraud = kde_fraud(x_grid)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(x_grid, pdf_normal, color="green", alpha=0.30,
                    label=f"Normal (n={len(normal):,})")
    ax.plot(x_grid, pdf_normal, color="green", lw=2)
    ax.fill_between(x_grid, pdf_fraud, color="red", alpha=0.30,
                    label=f"Fraud (n={len(fraud):,})")
    ax.plot(x_grid, pdf_fraud, color="red", lw=1.4)

    ax.set_xlim(0, 22)
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Reconstruction Error — Normal vs Fraud")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / "02_density_separation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
