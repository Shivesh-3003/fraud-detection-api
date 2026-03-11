import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
INPUT_FILE = 'X_train_AE.csv'
MODEL_SAVE_PATH = 'autoencoder_model.pth'

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
    print(f"   - Sample Count: {len(df_train)}")

    # Convert to Tensors
    X_train_tensor = torch.tensor(df_train.values, dtype=torch.float32).to(DEVICE)
    
    # Create DataLoader
    dataset = TensorDataset(X_train_tensor, X_train_tensor) # Target is same as Input
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize Model
    model = Autoencoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss() # Measures reconstruction error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    print(f"\n[STEP 2] Starting Training ({EPOCHS} Epochs)...")
    history = {'loss': []}

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
        
        # Average loss for the epoch
        avg_loss = train_loss / len(dataloader)
        history['loss'].append(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.6f}")

    # 4. Save Model
    print("\n[STEP 3] Saving Model Artifacts")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✓ Saved Model: {MODEL_SAVE_PATH}")

    # 5. Plot Training Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history['loss'], label='Training Loss', color='blue')
    plt.title('Autoencoder Training Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig('06_ae_training_loss.png')
    print("✓ Saved Training Plot: 06_ae_training_loss.png")
    
    print("\n✅ TRAINING COMPLETE.")

if __name__ == "__main__":
    train()