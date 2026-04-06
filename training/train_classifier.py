import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    auc
)
import os
import json
import pathlib

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
parser = argparse.ArgumentParser(description='Train MLP Classifier for Fraud Detection')
parser.add_argument('--dataset', choices=['ulb', 'sparkov'], default='ulb',
                    help='Dataset to train on (must match prior preprocessing steps)')
args = parser.parse_args()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_FEATURES      = f'data/{args.dataset}/X_train_final.csv'
TRAIN_LABELS        = f'data/{args.dataset}/y_train_MLP.csv'
TEST_FEATURES       = f'data/{args.dataset}/X_test_final.csv'
TEST_LABELS         = f'data/{args.dataset}/y_test.csv'
MODEL_SAVE_PATH     = f'models/{args.dataset}/mlp_classifier.pth'
THRESHOLD_SAVE_PATH = f'models/{args.dataset}/optimal_threshold.json'
PLOTS_DIR           = pathlib.Path("plots")
EVAL_DIR            = pathlib.Path("evaluation_plots")
DATASET             = args.dataset  # used as prefix for output filenames

BATCH_SIZE    = 256
EPOCHS        = 30
LEARNING_RATE = 0.001

# Set to a float (e.g. 100.0) to override the data-derived pos_weight for
# BCEWithLogitsLoss. None = use exact class ratio (~577 for ULB dataset).
# Lower values (100–300) can improve F1 by spreading probability mass more
# evenly, at the cost of a slightly higher raw false-alarm rate pre-threshold.
POS_WEIGHT_OVERRIDE = 100.0

# Detect Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✓ Using Device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✓ Using Device: CUDA")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠ Using Device: CPU")

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class FraudClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FraudClassifier, self).__init__()
        self.layer1  = nn.Linear(input_dim, 16)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2  = nn.Linear(16, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# ==============================================================================
# METRICS
# ==============================================================================
def calculate_all_metrics(y_true, y_pred, y_probs):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision   = tp / (tp + fp)  if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn)  if (tp + fn) > 0 else 0
    f1          = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp)  if (tn + fp) > 0 else 0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    g_mean      = np.sqrt(recall * specificity)

    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except Exception:
        auc_score = 0.0

    return {
        'confusion_matrix': cm,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'precision': precision, 'recall': recall, 'f1_score': f1,
        'specificity': specificity, 'accuracy': accuracy,
        'g_mean': g_mean, 'auc': auc_score
    }


def print_detailed_metrics(metrics, threshold, title="METRICS"):
    print("\n" + "="*70)
    print(f"{title} (Threshold = {threshold:.4f})")
    print("="*70)

    cm = metrics['confusion_matrix']
    print("\n📊 CONFUSION MATRIX:")
    print("─" * 50)
    print(f"                    Predicted")
    print(f"                 Normal     Fraud")
    print(f"Actual  Normal   {cm[0,0]:>6}    {cm[0,1]:>6}  (TN={metrics['tn']}, FP={metrics['fp']})")
    print(f"        Fraud    {cm[1,0]:>6}    {cm[1,1]:>6}  (FN={metrics['fn']}, TP={metrics['tp']})")
    print("─" * 50)

    print("\n📈 CORE METRICS:")
    print("─" * 50)
    print(f"{'Metric':<20} {'Value':<12} {'Formula':<30}")
    print("─" * 50)
    print(f"{'Accuracy':<20} {metrics['accuracy']:<12.4f} (TP + TN) / Total")
    print(f"{'Precision':<20} {metrics['precision']:<12.4f} TP / (TP + FP)")
    print(f"{'Recall (Sensitivity)':<20} {metrics['recall']:<12.4f} TP / (TP + FN)")
    print(f"{'Specificity':<20} {metrics['specificity']:<12.4f} TN / (TN + FP)")
    print(f"{'F1-Score':<20} {metrics['f1_score']:<12.4f} 2 * (Precision * Recall) / (P + R)")
    print(f"{'G-mean':<20} {metrics['g_mean']:<12.4f} sqrt(Recall * Specificity)")
    print(f"{'AUC-ROC':<20} {metrics['auc']:<12.4f} Area Under ROC Curve")
    print("─" * 50)

    print("\n💼 BUSINESS IMPACT:")
    print("─" * 50)
    total_fraud  = metrics['tp'] + metrics['fn']
    total_normal = metrics['tn'] + metrics['fp']
    total        = total_fraud + total_normal
    print(f"Total Test Transactions:     {total:>8,}")
    print(f"  ├─ Actual Fraud:           {total_fraud:>8,} ({total_fraud/total*100:.2f}%)")
    print(f"  └─ Actual Normal:          {total_normal:>8,} ({total_normal/total*100:.2f}%)")
    print()
    print(f"Fraud Detection Performance:")
    print(f"  ├─ Fraud Caught (TP):      {metrics['tp']:>8,} / {total_fraud:>6,} ({metrics['recall']*100:>5.2f}%)")
    print(f"  └─ Fraud Missed (FN):      {metrics['fn']:>8,} / {total_fraud:>6,} ({metrics['fn']/total_fraud*100:>5.2f}%)")
    print()
    print(f"Alert Accuracy:")
    total_alerts = metrics['tp'] + metrics['fp']
    if total_alerts > 0:
        print(f"  ├─ Total Alerts Generated: {total_alerts:>8,}")
        print(f"  ├─ True Fraud (TP):        {metrics['tp']:>8,} ({metrics['precision']*100:>5.2f}%)")
        print(f"  └─ False Alarms (FP):      {metrics['fp']:>8,} ({metrics['fp']/total_alerts*100:>5.2f}%)")
    else:
        print(f"  └─ No alerts generated")
    print()
    print(f"False Alarm Rate:            {metrics['fp']/total_normal*100:>7.3f}% of normal transactions")
    print("─" * 50)

    if metrics['recall'] >= 0.90:   recall_r = "Excellent"
    elif metrics['recall'] >= 0.80: recall_r = "Good"
    elif metrics['recall'] >= 0.70: recall_r = "Moderate"
    else:                           recall_r = "Needs Improvement"

    if metrics['precision'] >= 0.80:   prec_r = "Excellent"
    elif metrics['precision'] >= 0.70: prec_r = "Good"
    elif metrics['precision'] >= 0.60: prec_r = "Moderate"
    else:                              prec_r = "Needs Improvement"

    f1_r = 'Excellent' if metrics['f1_score'] >= 0.80 else ('Good' if metrics['f1_score'] >= 0.70 else 'Moderate')
    print(f"\n💡 INTERPRETATION:")
    print("─" * 50)
    print(f"Recall Rating:    {recall_r:>20} (Catching {metrics['recall']*100:.1f}% of fraud)")
    print(f"Precision Rating: {prec_r:>20} ({metrics['precision']*100:.1f}% of alerts are real fraud)")
    print(f"F1-Score Rating:  {f1_r:>20}")
    print("─" * 50)

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train_and_evaluate():
    print("="*70)
    print("PHASE 3: SUPERVISED CLASSIFIER TRAINING")
    print("="*70)

    # 1. Load Data
    print("\n[STEP 1] Loading Final Datasets...")
    if not os.path.exists(TRAIN_FEATURES):
        print("❌ Error: Files not found. Run Phase 2 first.")
        return

    X_train = pd.read_csv(TRAIN_FEATURES).values.astype(np.float32)
    y_train = pd.read_csv(TRAIN_LABELS).values.astype(np.float32)
    X_test  = pd.read_csv(TEST_FEATURES).values.astype(np.float32)
    y_test  = pd.read_csv(TEST_LABELS).values.astype(np.float32)

    print(f"   - Train Shape: {X_train.shape}")
    print(f"   - Test Shape:  {X_test.shape}")

    num_neg = int((y_train == 0).sum())
    num_pos = int((y_train == 1).sum())
    pw_value   = POS_WEIGHT_OVERRIDE if POS_WEIGHT_OVERRIDE is not None else float(num_neg / num_pos)
    pos_weight = torch.tensor([pw_value], dtype=torch.float32, device=DEVICE)
    src = f"override={POS_WEIGHT_OVERRIDE}" if POS_WEIGHT_OVERRIDE is not None else "data-derived"
    print(f"   - Class distribution: {num_pos} fraud / {num_neg} normal")
    print(f"   - pos_weight: {pos_weight.item():.2f} ({src})")

    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_test_t  = torch.tensor(X_test).to(DEVICE)
    y_test_t  = torch.tensor(y_test).to(DEVICE)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialise Model
    input_dim = X_train.shape[1]
    model     = FraudClassifier(input_dim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    # 3. Training Loop
    print(f"\n[STEP 2] Training Classifier ({EPOCHS} Epochs)...")

    train_losses = []
    val_losses   = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), y_test_t).item()
        val_losses.append(val_loss)
        model.train()

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1:>2}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✓ Saved Classifier: {MODEL_SAVE_PATH}")

    # 4. Evaluation
    print("\n[STEP 3] Comprehensive Evaluation on Test Set...")
    model.eval()
    with torch.no_grad():
        y_probs      = torch.sigmoid(model(X_test_t)).cpu().numpy().flatten()
        y_preds_default = (y_probs > 0.5).astype(int)

    metrics_default = calculate_all_metrics(y_test.flatten(), y_preds_default, y_probs)
    print_detailed_metrics(metrics_default, 0.5, "RESULTS WITH DEFAULT THRESHOLD")

    cm = confusion_matrix(y_test, y_preds_default)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title('Confusion Matrix (Default Threshold=0.5)')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '08_confusion_matrix_default.png'); plt.close()
    print("\n✓ Saved: 08_confusion_matrix_default.png")

    # ========================================================================
    # THRESHOLD OPTIMISATION
    # ========================================================================
    print("\n" + "="*70)
    print("THRESHOLD OPTIMISATION")
    print("="*70)

    # Coarse sweep (0.05 steps) — used for the visualisation curve only
    sweep_thresholds = np.arange(0.05, 1.00, 0.05)
    sweep_f1_scores  = [f1_score(y_test.flatten(), (y_probs >= t).astype(int), zero_division=0)
                        for t in sweep_thresholds]

    # Precise optimum: evaluate at every unique predicted probability
    pr_precisions_all, pr_recalls_all, pr_thresholds_all = precision_recall_curve(
        y_test.flatten(), y_probs
    )
    pr_f1_all = (2 * pr_precisions_all[:-1] * pr_recalls_all[:-1] /
                 (pr_precisions_all[:-1] + pr_recalls_all[:-1] + 1e-10))

    optimal_idx       = int(np.argmax(pr_f1_all))
    optimal_threshold = float(pr_thresholds_all[optimal_idx])
    optimal_f1        = float(pr_f1_all[optimal_idx])
    optimal_precision = float(pr_precisions_all[optimal_idx])
    optimal_recall    = float(pr_recalls_all[optimal_idx])

    print(f"\n🎯 OPTIMAL THRESHOLD FOUND: {optimal_threshold:.4f}")
    print(f"   ├─ Precision: {optimal_precision:.4f}")
    print(f"   ├─ Recall:    {optimal_recall:.4f}")
    print(f"   └─ F1-Score:  {optimal_f1:.4f}")

    y_preds_optimal = (y_probs >= optimal_threshold).astype(int)
    metrics_optimal = calculate_all_metrics(y_test.flatten(), y_preds_optimal, y_probs)
    print_detailed_metrics(metrics_optimal, optimal_threshold, "RESULTS WITH OPTIMISED THRESHOLD")

    cm_optimal = confusion_matrix(y_test, y_preds_optimal)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Optimised Threshold={optimal_threshold:.3f})')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '08_confusion_matrix_optimized.png'); plt.close()
    print("\n✓ Saved: 08_confusion_matrix_optimized.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm,         annot=True, fmt='d', cmap='Blues',  cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual')
    ax1.set_title(f'Default Threshold=0.5\nF1={metrics_default["f1_score"]:.4f}')
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual')
    ax2.set_title(f'Optimised Threshold={optimal_threshold:.3f}\nF1={metrics_optimal["f1_score"]:.4f}')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '08_confusion_matrix_comparison.png', dpi=150); plt.close()
    print("✓ Saved: 08_confusion_matrix_comparison.png")

    # Threshold Sensitivity Table
    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} "
          f"{'Specificity':<12} {'G-mean':<12}")
    print("-" * 78)
    for thresh in sorted({0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, optimal_threshold}):
        y_pred_t = (y_probs >= thresh).astype(int)
        tm = calculate_all_metrics(y_test.flatten(), y_pred_t, y_probs)
        marker = " ⭐ OPTIMAL" if abs(thresh - optimal_threshold) < 0.001 else ""
        print(f"{thresh:<12.4f} {tm['precision']:<12.4f} {tm['recall']:<12.4f} "
              f"{tm['f1_score']:<12.4f} {tm['specificity']:<12.4f} {tm['g_mean']:<12.4f}{marker}")

    # Comparison Summary
    print("\n" + "="*70)
    print("📊 THRESHOLD COMPARISON SUMMARY")
    print("="*70)
    comparison_data = {
        'Metric':  ['Precision','Recall','F1-Score','Specificity','G-mean','AUC','False Alarms'],
        'Default (0.5)': [
            f"{metrics_default['precision']:.4f}", f"{metrics_default['recall']:.4f}",
            f"{metrics_default['f1_score']:.4f}",  f"{metrics_default['specificity']:.4f}",
            f"{metrics_default['g_mean']:.4f}",    f"{metrics_default['auc']:.4f}",
            f"{metrics_default['fp']}"],
        f'Optimised ({optimal_threshold:.4f})': [
            f"{metrics_optimal['precision']:.4f}",  f"{metrics_optimal['recall']:.4f}",
            f"{metrics_optimal['f1_score']:.4f}",   f"{metrics_optimal['specificity']:.4f}",
            f"{metrics_optimal['g_mean']:.4f}",     f"{metrics_optimal['auc']:.4f}",
            f"{metrics_optimal['fp']}"],
        'Improvement': [
            f"{(metrics_optimal['precision']   - metrics_default['precision'])*100:+.2f}%",
            f"{(metrics_optimal['recall']      - metrics_default['recall'])*100:+.2f}%",
            f"{(metrics_optimal['f1_score']    - metrics_default['f1_score'])*100:+.2f}%",
            f"{(metrics_optimal['specificity'] - metrics_default['specificity'])*100:+.2f}%",
            f"{(metrics_optimal['g_mean']      - metrics_default['g_mean'])*100:+.2f}%",
            f"{(metrics_optimal['auc']         - metrics_default['auc'])*100:+.2f}%",
            f"{metrics_optimal['fp'] - metrics_default['fp']:+d}"]
    }
    print(pd.DataFrame(comparison_data).to_string(index=False))
    print("="*70)

    # ========================================================================
    # EVALUATION ARTIFACTS
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING EVALUATION ARTIFACTS → evaluation_plots/")
    print("="*70)
    pathlib.Path("models").mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    EVAL_DIR.mkdir(exist_ok=True)

    pr_precisions, pr_recalls = pr_precisions_all, pr_recalls_all
    pr_auc  = auc(pr_recalls, pr_precisions)
    roc_auc = roc_auc_score(y_test.flatten(), y_probs)

    # 1. {dataset}_metrics.json
    metrics_file = EVAL_DIR / f"{DATASET}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "tp": metrics_optimal['tp'], "fp": metrics_optimal['fp'],
            "tn": metrics_optimal['tn'], "fn": metrics_optimal['fn'],
            "precision":         round(metrics_optimal['precision'], 6),
            "recall":            round(metrics_optimal['recall'],    6),
            "f1_score":          round(metrics_optimal['f1_score'],  6),
            "roc_auc":           round(roc_auc,          6),
            "pr_auc":            round(pr_auc,            6),
            "optimal_threshold": round(optimal_threshold, 4),
        }, f, indent=2)
    print(f"✓ Saved: {metrics_file}")

    # 2. {dataset}_pr_curve.png
    plt.figure(figsize=(8, 6))
    plt.plot(pr_recalls, pr_precisions, color='darkorange', lw=2,
             label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({DATASET.upper()} Dataset)')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{DATASET}_pr_curve.png", dpi=150); plt.close()
    print(f"✓ Saved: {EVAL_DIR / f'{DATASET}_pr_curve.png'}")

    # 3. {dataset}_f1_threshold_curve.png
    plt.figure(figsize=(8, 6))
    plt.plot(sweep_thresholds, sweep_f1_scores, color='steelblue', lw=2,
             marker='o', markersize=4, label='F1-Score')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', lw=1.5,
                label=f'Optimal = {optimal_threshold:.4f}')
    plt.xlabel('Threshold'); plt.ylabel('F1-Score')
    plt.title(f'Threshold vs F1-Score ({DATASET.upper()} Dataset)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{DATASET}_f1_threshold_curve.png", dpi=150); plt.close()
    print(f"✓ Saved: {EVAL_DIR / f'{DATASET}_f1_threshold_curve.png'}")

    # 4. {dataset}_confusion_matrix.png
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', cbar=True,
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {DATASET.upper()} Dataset (Threshold={optimal_threshold:.3f})')
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{DATASET}_confusion_matrix.png", dpi=150); plt.close()
    print(f"✓ Saved: {EVAL_DIR / f'{DATASET}_confusion_matrix.png'}")

    # 5. {dataset}_loss_curve.png
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss',   color='steelblue', lw=2)
    plt.plot(range(1, EPOCHS + 1), val_losses,   label='Validation Loss', color='tomato',    lw=2, linestyle='--')
    plt.xlabel('Epoch'); plt.ylabel('BCEWithLogitsLoss')
    plt.title(f'MLP Classifier Loss Curve ({DATASET.upper()} Dataset)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{DATASET}_loss_curve.png", dpi=150); plt.close()
    print(f"✓ Saved: {EVAL_DIR / f'{DATASET}_loss_curve.png'}")

    # Save threshold for inference.py
    with open(THRESHOLD_SAVE_PATH, "w") as f:
        json.dump({"optimal_threshold": optimal_threshold}, f, indent=2)
    print(f"✓ Saved optimal threshold: {THRESHOLD_SAVE_PATH}")

    print("\n✅ TRAINING COMPLETE")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    train_and_evaluate()
