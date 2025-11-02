"""
🧠 AdaCore - Simple Neural Network for Ada's "Brain Seed"
A small PyTorch MLP designed for macOS local development
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

class AdaCore(nn.Module):
    """Small PyTorch MLP as Ada's neural core foundation"""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for better training stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
    def save_model(self, save_path: str):
        """Save model weights to specified path"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"💾 AdaCore model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """Load model weights from specified path"""
        if Path(load_path).exists():
            self.load_state_dict(torch.load(load_path, map_location=self._get_device()))
            print(f"🧠 AdaCore model loaded from: {load_path}")
            return True
        else:
            print(f"⚠️ Model file not found: {load_path}")
            return False
    
    def _get_device(self):
        """Auto-detect best available device (MPS for Apple Silicon, CUDA, or CPU)"""
        if torch.backends.mps.is_available():
            print("🍎 Using Apple Silicon (MPS) GPU acceleration")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("🚀 Using CUDA GPU acceleration")
            return torch.device("cuda")
        else:
            print("💻 Using CPU")
            return torch.device("cpu")

def create_ada_core(input_dim=512, hidden_dim=256, output_dim=512):
    """Factory function to create and initialize AdaCore"""
    model = AdaCore(input_dim, hidden_dim, output_dim)
    device = model._get_device()
    model.to(device)
    return model

def test_mps_support():
    """Test script to verify MPS GPU support"""
    print("🔍 Testing MPS (Metal Performance Shaders) support...")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        # Test a simple operation on MPS
        x = torch.rand(3, 3).to("mps")
        y = torch.mm(x, x.t())
        print("✅ MPS test successful - GPU acceleration ready!")
    else:
        print("❌ MPS not available - will use CPU")

if __name__ == "__main__":
    # Test the AdaCore model
    test_mps_support()
    
    print("\n🧪 Testing AdaCore neural network...")
    ada = create_ada_core()
    
    # Test with random input
    test_input = torch.randn(1, 512)
    output = ada(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ AdaCore test successful!")
    
    # Save test model
    ada.save_model("ada/storage/checkpoints/ada_core.pt")