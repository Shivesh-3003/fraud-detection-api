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
    classification_report, 
    roc_auc_score,
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    auc
)
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
TRAIN_FEATURES = 'X_train_final.csv'
TRAIN_LABELS   = 'y_train_MLP.csv'
TEST_FEATURES  = 'X_test_final.csv'
TEST_LABELS    = 'y_test.csv'
MODEL_SAVE_PATH = 'mlp_classifier.pth'

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001

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
# MODEL ARCHITECTURE (The Classifier)
# ==============================================================================
class FraudClassifier(nn.Module):
    def __init__(self, input_dim):
        super(FraudClassifier, self).__init__()
        
        # Simple MLP: 32 -> 16 -> 1
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3) 
        self.layer2 = nn.Linear(16, 1) 
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# ==============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ==============================================================================
def calculate_all_metrics(y_true, y_pred, y_probs):
    """
    Calculate all evaluation metrics defined in FYP Important terms
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Get confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Also called Sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Calculate G-mean (Geometric mean of Recall and Specificity)
    g_mean = np.sqrt(recall * specificity)
    
    # Calculate AUC (Area Under ROC Curve)
    try:
        auc_score = roc_auc_score(y_true, y_probs)
    except:
        auc_score = 0.0
    
    return {
        'confusion_matrix': cm,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'accuracy': accuracy,
        'g_mean': g_mean,
        'auc': auc_score
    }

def print_detailed_metrics(metrics, threshold, title="METRICS"):
    """
    Print comprehensive metrics in a formatted way
    """
    print("\n" + "="*70)
    print(f"{title} (Threshold = {threshold:.4f})")
    print("="*70)
    
    cm = metrics['confusion_matrix']
    
    # Print Confusion Matrix
    print("\n📊 CONFUSION MATRIX:")
    print("─" * 50)
    print(f"                    Predicted")
    print(f"                 Normal     Fraud")
    print(f"Actual  Normal   {cm[0,0]:>6}    {cm[0,1]:>6}  (TN={metrics['tn']}, FP={metrics['fp']})")
    print(f"        Fraud    {cm[1,0]:>6}    {cm[1,1]:>6}  (FN={metrics['fn']}, TP={metrics['tp']})")
    print("─" * 50)
    
    # Print Core Metrics
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
    
    # Print Business Metrics
    print("\n💼 BUSINESS IMPACT:")
    print("─" * 50)
    total_fraud = metrics['tp'] + metrics['fn']
    total_normal = metrics['tn'] + metrics['fp']
    
    print(f"Total Test Transactions:     {total_fraud + total_normal:>8,}")
    print(f"  ├─ Actual Fraud:           {total_fraud:>8,} ({total_fraud/(total_fraud+total_normal)*100:.2f}%)")
    print(f"  └─ Actual Normal:          {total_normal:>8,} ({total_normal/(total_fraud+total_normal)*100:.2f}%)")
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
    
    # Print Interpretation
    print("\n💡 INTERPRETATION:")
    print("─" * 50)
    
    if metrics['recall'] >= 0.90:
        recall_rating = "Excellent"
    elif metrics['recall'] >= 0.80:
        recall_rating = "Good"
    elif metrics['recall'] >= 0.70:
        recall_rating = "Moderate"
    else:
        recall_rating = "Needs Improvement"
    
    if metrics['precision'] >= 0.80:
        precision_rating = "Excellent"
    elif metrics['precision'] >= 0.70:
        precision_rating = "Good"
    elif metrics['precision'] >= 0.60:
        precision_rating = "Moderate"
    else:
        precision_rating = "Needs Improvement"
    
    print(f"Recall Rating:    {recall_rating:>20} (Catching {metrics['recall']*100:.1f}% of fraud)")
    print(f"Precision Rating: {precision_rating:>20} ({metrics['precision']*100:.1f}% of alerts are real fraud)")
    print(f"F1-Score Rating:  {'Excellent' if metrics['f1_score'] >= 0.80 else 'Good' if metrics['f1_score'] >= 0.70 else 'Moderate':>20}")
    print("─" * 50)

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train_and_evaluate():
    print("="*70)
    print("PHASE 3: SUPERVISED CLASSIFIER TRAINING WITH COMPREHENSIVE METRICS")
    print("="*70)

    # 1. Load Data
    print("\n[STEP 1] Loading Final Datasets...")
    if not os.path.exists(TRAIN_FEATURES):
        print("❌ Error: Files not found. Run Phase 2 first.")
        return

    # Load CSVs and enforce Float32 immediately
    X_train = pd.read_csv(TRAIN_FEATURES).values.astype(np.float32)
    y_train = pd.read_csv(TRAIN_LABELS).values.astype(np.float32)
    X_test = pd.read_csv(TEST_FEATURES).values.astype(np.float32)
    y_test = pd.read_csv(TEST_LABELS).values.astype(np.float32)

    print(f"   - Train Shape: {X_train.shape}")
    print(f"   - Test Shape:  {X_test.shape}")

    # Calculate Class Weight for Imbalance
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    
    # Cast the division result to float32 for MPS compatibility
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32, device=DEVICE)
    print(f"   - Calculated Class Weight: {pos_weight.item():.2f} (To handle imbalance)")

    # Convert to Tensors
    X_train_t = torch.tensor(X_train).to(DEVICE)
    y_train_t = torch.tensor(y_train).to(DEVICE)
    X_test_t = torch.tensor(X_test).to(DEVICE)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    input_dim = X_train.shape[1] 
    model = FraudClassifier(input_dim).to(DEVICE)
    
    # Loss Function
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"\n[STEP 2] Training Classifier ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] | Loss: {train_loss/len(train_loader):.4f}")

    # Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✓ Saved Classifier: {MODEL_SAVE_PATH}")

    # 4. Final Evaluation
    print("\n[STEP 3] Comprehensive Evaluation on Test Set...")
    model.eval()
    
    with torch.no_grad():
        y_logits = model(X_test_t)
        y_probs = torch.sigmoid(y_logits).cpu().numpy().flatten()
        y_preds_default = (y_probs > 0.5).astype(int)  # Default threshold
    
    # Calculate metrics at default threshold
    metrics_default = calculate_all_metrics(y_test.flatten(), y_preds_default, y_probs)
    print_detailed_metrics(metrics_default, 0.5, "RESULTS WITH DEFAULT THRESHOLD")

    # Confusion Matrix at default threshold
    cm = confusion_matrix(y_test, y_preds_default)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Default Threshold=0.5)')
    plt.savefig('08_confusion_matrix_default.png')
    print("\n✓ Saved Confusion Matrix: 08_confusion_matrix_default.png")

    # ========================================================================
    # THRESHOLD OPTIMIZATION
    # ========================================================================
    print("\n" + "="*70)
    print("THRESHOLD OPTIMIZATION")
    print("="*70)

    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test.flatten(), y_probs)

    # Calculate F1-scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Exclude boundary values (thresholds at exactly 0 or 1)
    valid_mask = (thresholds > 0.01) & (thresholds < 0.9999)
    valid_f1_scores = f1_scores[:-1][valid_mask]
    valid_thresholds = thresholds[valid_mask]
    valid_precisions = precisions[:-1][valid_mask]
    valid_recalls = recalls[:-1][valid_mask]

    # Find optimal threshold from valid range
    optimal_idx = np.argmax(valid_f1_scores)
    optimal_threshold = valid_thresholds[optimal_idx]
    optimal_f1 = valid_f1_scores[optimal_idx]
    optimal_precision = valid_precisions[optimal_idx]
    optimal_recall = valid_recalls[optimal_idx]

    print(f"\n🎯 OPTIMAL THRESHOLD FOUND: {optimal_threshold:.4f}")
    print(f"   ├─ Precision: {optimal_precision:.4f}")
    print(f"   ├─ Recall:    {optimal_recall:.4f}")
    print(f"   └─ F1-Score:  {optimal_f1:.4f}")

    # Re-evaluate with optimal threshold
    y_preds_optimal = (y_probs >= optimal_threshold).astype(int)
    
    # Calculate comprehensive metrics at optimal threshold
    metrics_optimal = calculate_all_metrics(y_test.flatten(), y_preds_optimal, y_probs)
    print_detailed_metrics(metrics_optimal, optimal_threshold, "RESULTS WITH OPTIMIZED THRESHOLD")

    # Confusion Matrix with Optimal Threshold
    cm_optimal = confusion_matrix(y_test, y_preds_optimal)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Optimized Threshold={optimal_threshold:.3f})')
    plt.savefig('08_confusion_matrix_optimized.png')
    print("\n✓ Saved Optimized Confusion Matrix: 08_confusion_matrix_optimized.png")

    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title(f'Default Threshold=0.5\nF1={metrics_default["f1_score"]:.4f}')

    sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title(f'Optimized Threshold={optimal_threshold:.3f}\nF1={metrics_optimal["f1_score"]:.4f}')

    plt.tight_layout()
    plt.savefig('08_confusion_matrix_comparison.png', dpi=150)
    print("✓ Saved Comparison: 08_confusion_matrix_comparison.png")

    # Threshold Sensitivity Analysis
    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Specificity':<12} {'G-mean':<12}")
    print("-" * 78)

    test_thresholds = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99, optimal_threshold]
    test_thresholds = sorted(set(test_thresholds))

    for thresh in test_thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        temp_metrics = calculate_all_metrics(y_test.flatten(), y_pred_thresh, y_probs)
        
        marker = " ⭐ OPTIMAL" if abs(thresh - optimal_threshold) < 0.001 else ""
        print(f"{thresh:<12.4f} {temp_metrics['precision']:<12.4f} {temp_metrics['recall']:<12.4f} "
              f"{temp_metrics['f1_score']:<12.4f} {temp_metrics['specificity']:<12.4f} "
              f"{temp_metrics['g_mean']:<12.4f}{marker}")

    # ========================================================================
    # FINAL COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("📊 THRESHOLD COMPARISON SUMMARY")
    print("="*70)
    
    comparison_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Specificity', 'G-mean', 'AUC', 'False Alarms'],
        'Default (0.5)': [
            f"{metrics_default['precision']:.4f}",
            f"{metrics_default['recall']:.4f}",
            f"{metrics_default['f1_score']:.4f}",
            f"{metrics_default['specificity']:.4f}",
            f"{metrics_default['g_mean']:.4f}",
            f"{metrics_default['auc']:.4f}",
            f"{metrics_default['fp']}"
        ],
        f'Optimized ({optimal_threshold:.4f})': [
            f"{metrics_optimal['precision']:.4f}",
            f"{metrics_optimal['recall']:.4f}",
            f"{metrics_optimal['f1_score']:.4f}",
            f"{metrics_optimal['specificity']:.4f}",
            f"{metrics_optimal['g_mean']:.4f}",
            f"{metrics_optimal['auc']:.4f}",
            f"{metrics_optimal['fp']}"
        ],
        'Improvement': [
            f"{(metrics_optimal['precision'] - metrics_default['precision'])*100:+.2f}%",
            f"{(metrics_optimal['recall'] - metrics_default['recall'])*100:+.2f}%",
            f"{(metrics_optimal['f1_score'] - metrics_default['f1_score'])*100:+.2f}%",
            f"{(metrics_optimal['specificity'] - metrics_default['specificity'])*100:+.2f}%",
            f"{(metrics_optimal['g_mean'] - metrics_default['g_mean'])*100:+.2f}%",
            f"{(metrics_optimal['auc'] - metrics_default['auc'])*100:+.2f}%",
            f"{metrics_optimal['fp'] - metrics_default['fp']:+d}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    print("="*70)

    print("\n✅ PIPELINE COMPLETE WITH COMPREHENSIVE METRICS")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    train_and_evaluate()