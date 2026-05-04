"""
Credit Card Fraud Detection - EDA & Preprocessing
Final Year Project: High-Performance Real-Time Fraud Detection System
Methodology: Unsupervised Anomaly Detection (Autoencoder) + MLP

Usage:
    python credit_card_eda_preprocessing.py --dataset ulb
    python credit_card_eda_preprocessing.py --dataset sparkov
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import warnings
import os
import pathlib

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Fraud Detection Preprocessing Pipeline')
parser.add_argument('--dataset', choices=['ulb', 'sparkov'], default='ulb',
                    help='Dataset to preprocess: ulb (European credit card) or sparkov (synthetic named features)')
args = parser.parse_args()

# Dataset-aware output directories
PLOTS_DIR  = pathlib.Path("plots")
DATA_DIR   = pathlib.Path(f"data/{args.dataset}")
MODELS_DIR = pathlib.Path(f"models/{args.dataset}")
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print(f"FYP PIPELINE: EDA & PREPROCESSING — {args.dataset.upper()} DATASET")
print("="*80)

# ============================================================================
# ULB (EUROPEAN CREDIT CARD) DATASET
# ============================================================================
def _run_ulb():
    """Preprocess the ULB (European credit card) dataset with anonymous V1-V28 features."""

    print("\n[SECTION 1] DATA LOADING")

    filename = 'data/creditcard.csv'
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

    print("  > Saving imbalance plot...")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df, palette=['#2ecc71', '#e74c3c'])
    plt.title('Class Distribution (The "Needle in a Haystack")', fontsize=14)
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.savefig(PLOTS_DIR / '01_imbalance_check.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved '01_imbalance_check.png'")

    # ============================================================================
    # SECTION 3: FEATURE ENGINEERING & CLEANING
    # ============================================================================
    print("\n[SECTION 3] FEATURE ENGINEERING")

    df['Amount_Log'] = np.log1p(df['Amount'])
    print("✓ Applied Log Transformation to 'Amount'")

    df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600)) % 24
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    print("✓ Engineered Cyclic Time Features (Hour_sin, Hour_cos)")

    df = df.drop(['Time', 'Amount', 'Hour'], axis=1)
    print(f"✓ Dropped raw columns. New shape: {df.shape}")

    # ============================================================================
    # SECTION 3.5: VISUAL VERIFICATION
    # ============================================================================
    print("\n[SECTION 3.5] VISUAL VERIFICATION")

    plt.figure(figsize=(8, 6))
    sns.histplot(df['Amount_Log'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Log-Transformed Amount')
    plt.xlabel('Log(Amount)')
    plt.ylabel('Count')
    plt.savefig(PLOTS_DIR / '02_log_amount_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved '02_log_amount_distribution.png'")

    plt.figure(figsize=(6, 6))
    plt.scatter(df['Hour_sin'], df['Hour_cos'], alpha=0.1, s=1, c='green')
    plt.title('Cyclic Time Check (Should be a Circle)')
    plt.xlabel('Hour_sin')
    plt.ylabel('Hour_cos')
    plt.axis('equal')
    plt.savefig(PLOTS_DIR / '03_cyclic_time_check.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved '03_cyclic_time_check.png'")

    plt.figure(figsize=(20, 16))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Full Correlation Matrix (Checking PCA Orthogonality)', fontsize=16)
    plt.savefig(PLOTS_DIR / '04_full_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> Saved '04_full_correlation_matrix.png'")

    plt.figure(figsize=(12, 8))
    corr_with_class = corr_matrix['Class'].drop('Class').sort_values(ascending=False)
    colors = ['red' if x > 0 else 'blue' for x in corr_with_class]
    corr_with_class.plot(kind='bar', color=colors)
    plt.title('Feature Correlation with Fraud (Class)', fontsize=16)
    plt.ylabel('Correlation Coefficient')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '05_correlation_with_class.png', dpi=300)
    plt.close()
    print("  -> Saved '05_correlation_with_class.png'")

    # ============================================================================
    # SECTION 4: SPLITTING (THE TWO-STAGE METHOD)
    # ============================================================================
    print("\n[SECTION 4] DATA SPLITTING (THE TWO-STAGE METHOD)")

    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    print(f"  - Global Train Shape: {X_train.shape}")
    print(f"  - Global Test Shape:  {X_test.shape}")

    # Extract Normal-Only Training Set for Autoencoder
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
    scaler.fit(X_train_normal)

    X_train_normal_scaled = scaler.transform(X_train_normal)
    X_train_scaled        = scaler.transform(X_train)
    X_test_scaled         = scaler.transform(X_test)

    print("✓ Data Scaled (Mean=0, Var=1) — Scaler fitted on normal transactions only")

    # ============================================================================
    # SECTION 6: SAVING ARTIFACTS
    # ============================================================================
    print("\n[SECTION 6] SAVING DATASETS")

    feature_cols = list(X.columns)  # V1-V28, Amount_Log, Hour_sin, Hour_cos

    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f"✓ Saved {MODELS_DIR / 'scaler.pkl'}")

    pd.DataFrame(X_train_normal_scaled, columns=feature_cols).to_csv(DATA_DIR / 'X_train_AE.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_train_AE.csv'}")

    pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv(DATA_DIR / 'X_train_MLP.csv', index=False)
    y_train.to_csv(DATA_DIR / 'y_train_MLP.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_train_MLP.csv'} & {DATA_DIR / 'y_train_MLP.csv'}")

    pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv(DATA_DIR / 'X_test.csv', index=False)
    y_test.to_csv(DATA_DIR / 'y_test.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_test.csv'} & {DATA_DIR / 'y_test.csv'}")

    # Save feature_config.json — consumed by ML service inference pipeline
    feature_config = {
        "dataset_type": "ulb",
        "feature_names": feature_cols,
        "ae_input_dim": len(feature_cols),
        "clf_input_dim": len(feature_cols) + 1,
        "has_ohe": False,
        "label_column": "Class"
    }
    with open(MODELS_DIR / 'feature_config.json', 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"✓ Saved {MODELS_DIR / 'feature_config.json'}")
    print(f"  ae_input_dim={feature_config['ae_input_dim']}, clf_input_dim={feature_config['clf_input_dim']}")

    print("\n" + "="*80)
    print("READY FOR PHASE 1: AUTOENCODER TRAINING")
    print(f"Next Step: python train_autoencoder.py --dataset ulb")
    print("="*80)


# ============================================================================
# SPARKOV (SYNTHETIC NAMED FEATURES) DATASET
# ============================================================================
def _run_sparkov():
    """
    Preprocess the Sparkov synthetic dataset with named, interpretable features.
    This dataset is ideal for SHAP explainability demonstrations — SHAP output will
    show meaningful names like 'category_shopping_net' instead of anonymous 'V14'.

    Feature engineering produces 20 features:
        Numeric (7):    amt_log, hour_sin, hour_cos, age, gender_M, city_pop_log, distance
        Categorical (13): category one-hot encoded (14 categories, drop first)
    """
    TRAIN_FILE = 'data/sparkov/fraudTrain.csv'
    TEST_FILE  = 'data/sparkov/fraudTest.csv'

    print("\n[SECTION 1] DATA LOADING (SPARKOV — PRE-SPLIT)")
    for path in [TRAIN_FILE, TEST_FILE]:
        if not os.path.exists(path):
            print(f"❌ ERROR: '{path}' not found.")
            print("   Download from: https://www.kaggle.com/datasets/kartik2112/fraud-detection")
            exit()

    df_train = pd.read_csv(TRAIN_FILE)
    df_test  = pd.read_csv(TEST_FILE)
    print(f"✓ Train set: {len(df_train):,} transactions")
    print(f"✓ Test set:  {len(df_test):,} transactions")
    print("  (Dataset is already split — no re-splitting needed)")

    # ============================================================================
    # SECTION 2: CLASS IMBALANCE CONFIRMATION
    # ============================================================================
    print("\n[SECTION 2] IMBALANCE ANALYSIS")

    train_fraud_pct = df_train['is_fraud'].mean() * 100
    test_fraud_pct  = df_test['is_fraud'].mean() * 100
    print(f"  Train — Legitimate: {(df_train['is_fraud']==0).sum():,} | Fraud: {(df_train['is_fraud']==1).sum():,} ({train_fraud_pct:.3f}%)")
    print(f"  Test  — Legitimate: {(df_test['is_fraud']==0).sum():,}  | Fraud: {(df_test['is_fraud']==1).sum():,} ({test_fraud_pct:.3f}%)")
    print(f"\n✅ CONFIRMED: Highly Imbalanced Dataset ({train_fraud_pct:.3f}% fraud in train)")

    plt.figure(figsize=(8, 6))
    counts = df_train['is_fraud'].value_counts()
    sns.barplot(x=['Normal', 'Fraud'], y=[counts[0], counts[1]], palette=['#2ecc71', '#e74c3c'])
    plt.title('Sparkov Class Distribution')
    plt.yscale('log')
    plt.ylabel('Count (Log Scale)')
    plt.savefig(PLOTS_DIR / 'sparkov_01_imbalance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved 'sparkov_01_imbalance.png'")

    # ============================================================================
    # SECTION 3: FEATURE ENGINEERING
    # ============================================================================
    print("\n[SECTION 3] FEATURE ENGINEERING (SPARKOV)")

    def _engineer_features(df):
        df = df.copy()
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['dob'] = pd.to_datetime(df['dob'])

        # Amount log transform (same principle as ULB's Amount_Log)
        df['amt_log'] = np.log1p(df['amt'])

        # Cyclic hour encoding (same as ULB)
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Customer age at time of transaction
        df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25

        # Gender as binary (1=Male, 0=Female)
        df['gender_M'] = (df['gender'] == 'M').astype(float)

        # City population (log-scaled to reduce skew)
        df['city_pop_log'] = np.log1p(df['city_pop'])

        # Geographic distance: customer location vs merchant location
        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 +
            (df['long'] - df['merch_long'])**2
        )

        return df

    df_train = _engineer_features(df_train)
    df_test  = _engineer_features(df_test)
    print("✓ Engineered: amt_log, hour_sin, hour_cos, age, gender_M, city_pop_log, distance")

    # ============================================================================
    # SECTION 3.1: CATEGORICAL ENCODING
    # ============================================================================
    print("\n[SECTION 3.1] CATEGORICAL ENCODING")

    # Fit OneHotEncoder on train only to avoid data leakage
    # sparse_output=False requires sklearn >= 1.2; use sparse=False for older versions
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    cat_train = ohe.fit_transform(df_train[['category']])
    cat_test  = ohe.transform(df_test[['category']])

    cat_feature_names = list(ohe.get_feature_names_out(['category']))
    all_categories    = sorted(df_train['category'].unique().tolist())

    print(f"✓ OHE 'category': {len(all_categories)} categories → {len(cat_feature_names)} dummy features (drop first)")
    print(f"  All categories: {all_categories}")
    print(f"  Dropped (first): {all_categories[0]}")

    # Validation: cyclic time plot
    plt.figure(figsize=(6, 6))
    plt.scatter(df_train['hour_sin'], df_train['hour_cos'], alpha=0.05, s=1, c='green')
    plt.title('Sparkov Cyclic Time Check (Should be a Circle)')
    plt.xlabel('hour_sin')
    plt.ylabel('hour_cos')
    plt.axis('equal')
    plt.savefig(PLOTS_DIR / 'sparkov_02_cyclic_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved 'sparkov_02_cyclic_time.png'")

    # Amount distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df_train['amt_log'], bins=50, kde=True, color='blue')
    plt.title('Sparkov Distribution of Log-Transformed Amount')
    plt.xlabel('log(Amount)')
    plt.savefig(PLOTS_DIR / 'sparkov_03_log_amount.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved 'sparkov_03_log_amount.png'")

    # ============================================================================
    # SECTION 4: ASSEMBLE FINAL FEATURE MATRICES
    # ============================================================================
    print("\n[SECTION 4] ASSEMBLING FEATURE MATRICES")

    NUMERIC_COLS = ['amt_log', 'hour_sin', 'hour_cos', 'age', 'gender_M', 'city_pop_log', 'distance']
    FEATURE_COLS = NUMERIC_COLS + cat_feature_names

    X_train = pd.concat([
        df_train[NUMERIC_COLS].reset_index(drop=True),
        pd.DataFrame(cat_train, columns=cat_feature_names)
    ], axis=1)
    y_train = df_train['is_fraud'].reset_index(drop=True)

    X_test = pd.concat([
        df_test[NUMERIC_COLS].reset_index(drop=True),
        pd.DataFrame(cat_test, columns=cat_feature_names)
    ], axis=1)
    y_test = df_test['is_fraud'].reset_index(drop=True)

    print(f"✓ Feature matrix: {len(FEATURE_COLS)} total features")
    print(f"  Numeric ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")
    print(f"  Categorical ({len(cat_feature_names)}): {cat_feature_names[:3]}... (and {len(cat_feature_names)-3} more)")

    # ============================================================================
    # SECTION 5: SCALING (FIT ON NORMALS ONLY)
    # ============================================================================
    print("\n[SECTION 5] SCALING — FIT ON NORMAL TRANSACTIONS ONLY")

    X_train_normal = X_train[y_train == 0]
    print(f"  Fitting scaler on {len(X_train_normal):,} normal transactions")
    print(f"  (This is CRITICAL: scaler must not learn from fraud patterns)")

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    X_train_normal_scaled = scaler.transform(X_train_normal)
    X_train_scaled        = scaler.transform(X_train)
    X_test_scaled         = scaler.transform(X_test)

    print("✓ Data Scaled (Mean=0, Var=1)")

    # ============================================================================
    # SECTION 6: SAVING ARTIFACTS
    # ============================================================================
    print("\n[SECTION 6] SAVING ARTIFACTS")

    # Scaler
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f"✓ Saved {MODELS_DIR / 'scaler.pkl'}")

    # OneHotEncoder (must be saved for identical inference-time encoding)
    joblib.dump(ohe, MODELS_DIR / 'onehot_encoder.pkl')
    print(f"✓ Saved {MODELS_DIR / 'onehot_encoder.pkl'}")

    # Datasets
    pd.DataFrame(X_train_normal_scaled, columns=FEATURE_COLS).to_csv(DATA_DIR / 'X_train_AE.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_train_AE.csv'} ({len(X_train_normal_scaled):,} rows, normal only)")

    pd.DataFrame(X_train_scaled, columns=FEATURE_COLS).to_csv(DATA_DIR / 'X_train_MLP.csv', index=False)
    y_train.rename('is_fraud').to_csv(DATA_DIR / 'y_train_MLP.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_train_MLP.csv'} & {DATA_DIR / 'y_train_MLP.csv'}")

    pd.DataFrame(X_test_scaled, columns=FEATURE_COLS).to_csv(DATA_DIR / 'X_test.csv', index=False)
    y_test.rename('is_fraud').to_csv(DATA_DIR / 'y_test.csv', index=False)
    print(f"✓ Saved {DATA_DIR / 'X_test.csv'} & {DATA_DIR / 'y_test.csv'}")

    # Feature config — key bridge between training and inference
    feature_config = {
        "dataset_type": "sparkov",
        "feature_names": FEATURE_COLS,
        "ae_input_dim": len(FEATURE_COLS),
        "clf_input_dim": len(FEATURE_COLS) + 1,
        "has_ohe": True,
        "label_column": "is_fraud"
    }
    with open(MODELS_DIR / 'feature_config.json', 'w') as f:
        json.dump(feature_config, f, indent=2)
    print(f"✓ Saved {MODELS_DIR / 'feature_config.json'}")
    print(f"  ae_input_dim={feature_config['ae_input_dim']}, clf_input_dim={feature_config['clf_input_dim']}")

    print("\n" + "="*80)
    print("READY FOR PHASE 1: AUTOENCODER TRAINING")
    print(f"Feature count: {len(FEATURE_COLS)} features ({len(FEATURE_COLS)+1} with Reconstruction_Error)")
    print(f"Next Step: python train_autoencoder.py --dataset sparkov")
    print("="*80)


# ============================================================================
# DISPATCH
# ============================================================================
if args.dataset == 'ulb':
    _run_ulb()
else:
    _run_sparkov()
