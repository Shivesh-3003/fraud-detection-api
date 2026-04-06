import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
parser = argparse.ArgumentParser(description='Train Autoencoder for Fraud Detection')
parser.add_argument('--dataset', choices=['ulb', 'sparkov'], default='ulb',
                    help='Dataset to train on (must match preprocessing output)')
args = parser.parse_args()

PLOTS_DIR  = pathlib.Path("plots")
MODELS_DIR = pathlib.Path(f"models/{args.dataset}")
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
INPUT_FILE = f'data/{args.dataset}/X_train_AE.csv'
MODEL_SAVE_PATH = str(MODELS_DIR / 'autoencoder_model.pth')
VAL_SPLIT = 0.10  # 90/10 train/val split

# Detect Device (Prioritize Apple Metal (MPS) > CUDA > CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✓ Using Device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✓ Using Device: CUDA (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠ Using Device: CPU (Slower)")

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder: Compresses the input (31 features -> 8 features)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 14),
            nn.Tanh(),
            nn.Linear(14, 8), # Bottleneck: Latent Representation
            nn.Tanh()
        )

        # Decoder: Reconstructs the input (8 features -> 31 features)
        self.decoder = nn.Sequential(
            nn.Linear(8, 14),
            nn.Tanh(),
            nn.Linear(14, 20),
            nn.Tanh(),
            nn.Linear(20, input_dim) # Output layer (No activation, let loss function handle range)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ==============================================================================
# TRAINING PIPELINE
# ==============================================================================
def train():
    print("="*60)
    print("PHASE 1: AUTOENCODER TRAINING")
    print("="*60)

    # 1. Load Data
    print(f"\n[STEP 1] Loading Training Data: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: {INPUT_FILE} not found. Run preprocessing first.")
        return

    df_train = pd.read_csv(INPUT_FILE)
    input_dim = df_train.shape[1]
    print(f"   - Input Dimensions: {input_dim} features")
    print(f"   - Total Samples: {len(df_train)}")

    # 90/10 train/val split (shuffle before splitting)
    df_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(df_shuffled) * VAL_SPLIT)
    df_val_data = df_shuffled.iloc[:val_size]
    df_train_data = df_shuffled.iloc[val_size:]
    print(f"   - Train Samples: {len(df_train_data)} | Val Samples: {len(df_val_data)}")

    # Convert to Tensors
    X_train_tensor = torch.tensor(df_train_data.values, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(df_val_data.values, dtype=torch.float32).to(DEVICE)

    # Create DataLoader (training only; val is evaluated in full each epoch)
    dataset = TensorDataset(X_train_tensor, X_train_tensor)  # Target is same as Input
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = Autoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()  # Measures reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"\n[STEP 2] Starting Training ({EPOCHS} Epochs)...")
    ae_train_losses = []
    ae_val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch_features, _ in dataloader:
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Average training loss for the epoch
        avg_train_loss = train_loss / len(dataloader)
        ae_train_losses.append(avg_train_loss)

        # Validation loss (no gradient)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            avg_val_loss = criterion(val_outputs, X_val_tensor).item()
        ae_val_losses.append(avg_val_loss)
        model.train()

        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # 4. Save Model
    print("\n[STEP 3] Saving Model Artifacts")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✓ Saved Model: {MODEL_SAVE_PATH}")

    # 5. Plot Training & Validation Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), ae_train_losses, label='Training Loss', color='blue', lw=2)
    plt.plot(range(1, EPOCHS + 1), ae_val_losses, label='Validation Loss', color='tomato',
             lw=2, linestyle='--')
    plt.title('Autoencoder Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_name = f'{args.dataset}_ae_training_loss.png'
    plt.savefig(PLOTS_DIR / plot_name)
    plt.close()
    print(f"✓ Saved Training Plot: {plot_name}")

    print("\n✅ TRAINING COMPLETE.")

if __name__ == "__main__":
    train()
