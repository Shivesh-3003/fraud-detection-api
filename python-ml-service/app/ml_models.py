"""
PyTorch Model Definitions for Fraud Detection

These architectures exactly match the training scripts.
Any mismatch will cause state_dict loading to fail.

Autoencoder Architecture (from train_autoencoder.py):
    Input (31) → 20 (Tanh) → 14 (Tanh) → 8 (Tanh) → 14 (Tanh) → 20 (Tanh) → Output (31)

MLP Classifier Architecture (from train_classifier.py):
    Input (32) → 16 (ReLU + Dropout 0.3) → 1 (Logits)
"""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder for anomaly detection via reconstruction error.
    
    Trained on NORMAL transactions only. Fraudulent transactions
    will have higher reconstruction error because the model hasn't
    learned to reconstruct their patterns.
    
    Architecture:
        Encoder: 31 → 20 → 14 → 8 (bottleneck)
        Decoder: 8 → 14 → 20 → 31
        
    Input: 31 scaled features (V1-V28, Amount_Log, Hour_sin, Hour_cos)
    Output: 31 reconstructed features
    """
    
    def __init__(self, input_dim: int = 31):
        super(Autoencoder, self).__init__()
        
        # Encoder: Compresses input to 8-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.Tanh(),
            nn.Linear(20, 14),
            nn.Tanh(),
            nn.Linear(14, 8),  # Bottleneck layer
            nn.Tanh()
        )
        
        # Decoder: Reconstructs input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(8, 14),
            nn.Tanh(),
            nn.Linear(14, 20),
            nn.Tanh(),
            nn.Linear(20, input_dim)  # No activation on output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate Mean Squared Error between input and reconstruction.
        
        Args:
            x: Input tensor of shape (batch_size, 31)
            
        Returns:
            Tensor of shape (batch_size,) with MSE per sample
        """
        reconstructed = self.forward(x)
        # MSE per sample: mean across features (dim=1)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class FraudClassifier(nn.Module):
    """
    MLP Classifier for final fraud prediction.
    
    Takes the original 31 features PLUS the reconstruction error
    from the Autoencoder (32 features total).
    
    Architecture:
        Input (32) → Linear(16) → ReLU → Dropout(0.3) → Linear(1) → Logits
        
    Note: Output is raw logits. Apply sigmoid for probability.
    """
    
    def __init__(self, input_dim: int = 32):
        super(FraudClassifier, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(16, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits.
        
        Args:
            x: Input tensor of shape (batch_size, 32)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fraud probability (applies sigmoid to logits).
        
        Args:
            x: Input tensor of shape (batch_size, 32)
            
        Returns:
            Probability tensor of shape (batch_size,)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs
