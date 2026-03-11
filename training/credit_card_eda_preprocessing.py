"""
Credit Card Fraud Detection - EDA & Preprocessing
Final Year Project: High-Performance Real-Time Fraud Detection System
Methodology: Unsupervised Anomaly Detection (Autoencoder) + MLP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
# Default figure size, can be overridden per plot
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("FYP PIPELINE: EDA & PREPROCESSING (IMBALANCED APPROACH)")
print("="*80)

# ============================================================================
# SECTION 1: DATA LOADING
# ============================================================================
print("\n[SECTION 1] DATA LOADING")

filename = 'creditcard.csv'

if not os.path.exists(filename):
    print(f"❌ ERROR: '{filename}' not found.")
    print("   Please download the original Kaggle dataset (284,807 rows).")
    exit()
else:
    df = pd.read_csv(filename)
    print(f"✓ Dataset loaded: {len(df):,} transactions")

# ============================================================================
# SECTION 2: CLASS IMBALANCE CONFIRMATION
# ============================================================================
print("\n[SECTION 2] IMBALANCE ANALYSIS")

class_counts = df['Class'].value_counts()
fraud_pct = (class_counts[1] / len(df)) * 100

print(f"  - Legitimate (0): {class_counts[0]:,} ({100-fraud_pct:.3f}%)")
print(f"  - Fraud (1):      {class_counts[1]:,} ({fraud_pct:.3f}%)")

if fraud_pct > 10:
    print("\n⚠️ WARNING: This dataset looks balanced (>10% fraud).")
    print("   Your project definition requires a HIGHLY IMBALANCED dataset.")
    exit()
else:
    print(f"\n✅ CONFIRMED: Highly Imbalanced Dataset ({fraud_pct:.3f}% fraud).")
    print("   This aligns with your Project Definition requirements.")

# Visualizing the Imbalance
print("  > Saving imbalance plot...")
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Class Distribution (The "Needle in a Haystack")', fontsize=14)
plt.yscale('log') # Log scale is essential here to see the fraud bar
plt.ylabel('Count (Log Scale)')
plt.savefig('01_imbalance_check.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved '01_imbalance_check.png'")

# ============================================================================
# SECTION 3: FEATURE ENGINEERING & CLEANING
# ============================================================================
print("\n[SECTION 3] FEATURE ENGINEERING")

# 1. Log Transform Amount
df['Amount_Log'] = np.log1p(df['Amount'])
print("✓ Applied Log Transformation to 'Amount'")

# 2. Time Engineering (Cyclic Features)
df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600)) % 24

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
print("✓ Engineered Cyclic Time Features (Hour_sin, Hour_cos)")

# 3. Drop Columns
df = df.drop(['Time', 'Amount', 'Hour'], axis=1)
print(f"✓ Dropped raw columns. New shape: {df.shape}")

# ============================================================================
# SECTION 3.5: VISUAL VERIFICATION (SAVING DISTINCT PLOTS)
# ============================================================================
print("\n[SECTION 3.5] VISUAL VERIFICATION")
print("  > Generating and saving transformation checks...")

# --- Plot 1: Log Amount Distribution ---
plt.figure(figsize=(8, 6)) # Create a new figure
sns.histplot(df['Amount_Log'], bins=50, kde=True, color='blue')
plt.title('Distribution of Log-Transformed Amount')
plt.xlabel('Log(Amount)')
plt.ylabel('Count')
plt.savefig('02_log_amount_distribution.png', dpi=300, bbox_inches='tight')
plt.close() # Close the figure to free memory
print("  -> Saved '02_log_amount_distribution.png'")

# --- Plot 2: Cyclic Features Check ---
plt.figure(figsize=(6, 6)) # Square figure for the circle
plt.scatter(df['Hour_sin'], df['Hour_cos'], alpha=0.1, s=1, c='green')
plt.title('Cyclic Time Check (Should be a Circle)')
plt.xlabel('Hour_sin')
plt.ylabel('Hour_cos')
plt.axis('equal') # Ensure the aspect ratio is square
plt.savefig('03_cyclic_time_check.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> Saved '03_cyclic_time_check.png'")

# --- Plot 3: Full Correlation Matrix (Structure Check) ---
plt.figure(figsize=(20, 16)) 
corr_matrix = df.corr()

# We turn annot=False because 30x30 numbers are unreadable. 
# We rely on color intensity.
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Full Correlation Matrix (Checking PCA Orthogonality)', fontsize=16)
plt.savefig('04_full_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  -> Saved '04_full_correlation_matrix.png'")

# --- Plot 4: Correlation with Class (Feature Importance) ---
# This is much more useful than the heatmap for finding top features
plt.figure(figsize=(12, 8))
# Drop Class vs Class correlation (which is 1.0)
corr_with_class = corr_matrix['Class'].drop('Class').sort_values(ascending=False)

colors = ['red' if x > 0 else 'blue' for x in corr_with_class]
corr_with_class.plot(kind='bar', color=colors)
plt.title('Feature Correlation with Fraud (Class)', fontsize=16)
plt.ylabel('Correlation Coefficient')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('05_correlation_with_class.png', dpi=300)
plt.close()
print("  -> Saved '05_correlation_with_class.png'")

# ============================================================================
# SECTION 4: SPLITTING (THE TWO-STAGE METHOD)
# ============================================================================
print("\n[SECTION 4] DATA SPLITTING (THE TWO-STAGE METHOD)")

X = df.drop('Class', axis=1)
y = df['Class']

# 1. Stratified Split (80% Train, 20% Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Further Split Train into Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
)

print(f"  - Global Train Shape: {X_train.shape}")
print(f"  - Global Test Shape:  {X_test.shape}")

# 3. CREATE "NORMAL ONLY" DATASET FOR AUTOENCODER
train_normal_indices = y_train[y_train == 0].index
X_train_normal = X_train.loc[train_normal_indices]

print(f"\n🎯 [CRITICAL] Autoencoder Training Set Created:")
print(f"   - Contains ONLY Normal transactions (Class 0)")
print(f"   - Shape: {X_train_normal.shape}")
print(f"   - Fraud count in this set: {(y_train.loc[train_normal_indices] == 1).sum()} (Must be 0)")

# ============================================================================
# SECTION 5: SCALING
# ============================================================================
print("\n[SECTION 5] SCALING")

scaler = StandardScaler()

# Fit ONLY on the Normal Training Data
scaler.fit(X_train_normal)

# Transform everything
X_train_normal_scaled = scaler.transform(X_train_normal)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Data Scaled (Mean=0, Var=1)")

# ============================================================================
# SECTION 6: SAVING ARTIFACTS
# ============================================================================
print("\n[SECTION 6] SAVING DATASETS")

# 1. Save Scaler
joblib.dump(scaler, 'scaler.pkl')
print("✓ Saved scaler.pkl")

# 2. Save Autoencoder Training Data (Normal Only)
pd.DataFrame(X_train_normal_scaled, columns=X.columns).to_csv('X_train_AE.csv', index=False)
print("✓ Saved X_train_AE.csv (Use this to train PyTorch Autoencoder)")

# 3. Save MLP Training Data (Mixed Data + Labels)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('X_train_MLP.csv', index=False)
y_train.to_csv('y_train_MLP.csv', index=False)
print("✓ Saved X_train_MLP.csv & y_train_MLP.csv (Use this for Classifier)")

# 4. Save Test Data
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print("✓ Saved X_test.csv & y_test.csv")

print("\n" + "="*80)
print("READY FOR PHASE 1: AUTOENCODER TRAINING")
print("Next Step: Run your PyTorch script using 'X_train_AE.csv'")
print("="*80)