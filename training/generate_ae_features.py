import argparse
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pathlib

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
parser = argparse.ArgumentParser(description='Generate Autoencoder Reconstruction Error Features')
parser.add_argument('--dataset', choices=['ulb', 'sparkov'], default='ulb',
                    help='Dataset to process (must match preprocessing and autoencoder training)')
args = parser.parse_args()

PLOTS_DIR = pathlib.Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = f'models/{args.dataset}/autoencoder_model.pth'
FILES = {
    'train': f'data/{args.dataset}/X_train_MLP.csv',
    'test':  f'data/{args.dataset}/X_test.csv'
}
OUTPUT_FILES = {
    'train': f'data/{args.dataset}/X_train_final.csv',
    'test':  f'data/{args.dataset}/X_test_final.csv'
}

# Load label column name from feature_config.json (Class for ULB, is_fraud for Sparkov)
_config_path = f'models/{args.dataset}/feature_config.json'
_LABEL_COL = 'Class'  # default (ULB)
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        _LABEL_COL = json.load(_f).get('label_column', 'Class')

# Detect Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==============================================================================
# DEFINE MODEL ARCHITECTURE
# ==============================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20), nn.Tanh(),
            nn.Linear(20, 14), nn.Tanh(),
            nn.Linear(14, 8), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 14), nn.Tanh(),
            nn.Linear(14, 20), nn.Tanh(),
            nn.Linear(20, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==============================================================================
# EXECUTION
# ==============================================================================
def generate_features():
    print("="*60)
    print("PHASE 2: FEATURE GENERATION (RECONSTRUCTION ERROR)")
    print("="*60)

    # 1. Load Model
    print(f"\n[STEP 1] Loading Trained Model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found.")
        return

    # We need to peek at the file to get input_dim
    temp_df = pd.read_csv(FILES['train'], nrows=1)
    input_dim = temp_df.shape[1]
    
    model = Autoencoder(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval() # Set to evaluation mode
    print("✓ Model loaded and frozen.")

    # 2. Process Files
    print(f"\n[STEP 2] Generating Reconstruction Errors...")
    
    for key, filepath in FILES.items():
        print(f"   > Processing {key.upper()} set: {filepath}...")
        df = pd.read_csv(filepath)
        
        # Convert to Tensor
        data_tensor = torch.tensor(df.values, dtype=torch.float32).to(DEVICE)
        
        # Forward Pass (No Gradient Calculation needed)
        with torch.no_grad():
            reconstructed = model(data_tensor)
            
            # Calculate MSE per row: (Input - Output)^2
            # We mean across the features (dim=1) to get one single error number per row
            mse_loss = torch.mean((data_tensor - reconstructed) ** 2, dim=1)
            
        # Move back to CPU and add to DataFrame
        error_values = mse_loss.cpu().numpy()
        df['Reconstruction_Error'] = error_values
        
        # Save
        save_path = OUTPUT_FILES[key]
        df.to_csv(save_path, index=False)
        print(f"     -> Saved with new feature: {save_path}")

    # 3. Validation Visualization
    print(f"\n[STEP 3] Validating Feature Separation...")
    
    # We need labels to visualize if it worked. Load y_train_MLP
    y_train = pd.read_csv(f'data/{args.dataset}/y_train_MLP.csv')
    df_train = pd.read_csv(OUTPUT_FILES['train'])
    df_train[_LABEL_COL] = y_train[_LABEL_COL]  # Temporary join for plotting

    plt.figure(figsize=(10, 6))
    
    # Plot Normal vs Fraud Reconstruction Errors
    sns.kdeplot(data=df_train[df_train[_LABEL_COL]==0], x='Reconstruction_Error',
                fill=True, color='green', label='Normal (0)', alpha=0.3)
    sns.kdeplot(data=df_train[df_train[_LABEL_COL]==1], x='Reconstruction_Error',
                fill=True, color='red', label='Fraud (1)', alpha=0.3)
    
    plt.title('Reconstruction Error Separation (The "Signal")')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.xlim(0, 5) # Zoom in to see the separation
    plt.legend()
    plot_name = f'{args.dataset}_error_separation.png'
    plt.savefig(PLOTS_DIR / plot_name)
    print(f"✓ Saved Separation Plot: {plot_name}")
    
    print("\n✅ PHASE 2 COMPLETE.")
    print("Next Step: Train the final MLP Classifier on 'X_train_final.csv'")

if __name__ == "__main__":
    generate_features()