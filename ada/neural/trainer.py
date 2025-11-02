"""
🏋️ Training Loop for Ada's Neural Network
Simple training implementation for Phase 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict
import os
from pathlib import Path
import json
import time

class AdaTrainer:
    """Simple trainer for Ada's neural network"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        # Ensure model is on the correct device
        self.device = model._get_device()
        self.model.to(self.device)  # Explicitly move model to device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss().to(self.device)  # Move criterion to device
        self.training_history = []
        
    def generate_dummy_data(self, num_samples=100, input_dim=512):
        """Generate dummy training data for testing"""
        inputs = torch.randn(num_samples, input_dim)
        # Create simple target function: output = input + noise
        targets = inputs + 0.1 * torch.randn(num_samples, input_dim)
        # Ensure both inputs and targets are on the same device as the model
        return inputs.to(self.device), targets.to(self.device)
    
    def train_epoch(self, train_loader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Ensure data and target are on the same device as the model
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return {"loss": avg_loss, "batches": num_batches}
    
    def train(self, epochs=5, batch_size=32, num_samples=100):
        """Main training loop"""
        print(f"🚀 Starting training for {epochs} epochs...")
        print(f"📊 Training on {num_samples} samples with batch size {batch_size}")
        
        # Generate training data
        train_inputs, train_targets = self.generate_dummy_data(num_samples)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            print(f"\n📈 Epoch {epoch + 1}/{epochs}")
            start_time = time.time()
            
            # Train for this epoch
            epoch_stats = self.train_epoch(train_loader)
            
            epoch_time = time.time() - start_time
            epoch_stats["epoch"] = epoch + 1
            epoch_stats["time"] = epoch_time
            
            self.training_history.append(epoch_stats)
            print(f"✅ Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"   Average Loss: {epoch_stats['loss']:.6f}")
        
        print("🎉 Training completed!")
        return self.training_history
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "model_config": {
                "input_dim": self.model.fc1.in_features,
                "hidden_dim": self.model.fc1.out_features,
                "output_dim": self.model.fc2.out_features
            }
        }
        
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.training_history = checkpoint.get("training_history", [])
            
            print(f"🧠 Checkpoint loaded from: {checkpoint_path}")
            print(f"📊 Training history: {len(self.training_history)} epochs")
            return True
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            return False
    
    def evaluate(self, test_inputs, test_targets):
        """Evaluate model on test data"""
        self.model.eval()
        with torch.no_grad():
            test_inputs = test_inputs.to(self.device)
            test_targets = test_targets.to(self.device)
            predictions = self.model(test_inputs)
            loss = self.criterion(predictions, test_targets)
        
        return {"test_loss": loss.item(), "predictions": predictions}

def run_quick_test():
    """Run a quick test of the training system"""
    print("🧪 Running quick training test...")
    
    # Import AdaCore
    from ada.neural.policy_network import AdaCore
    
    # Create model
    ada_core = AdaCore(input_dim=512, hidden_dim=256, output_dim=512)
    
    # Create trainer
    trainer = AdaTrainer(ada_core, learning_rate=0.01)
    
    # Quick training (2 epochs, small dataset)
    history = trainer.train(epochs=2, batch_size=16, num_samples=50)
    
    # Save model
    ada_core.save_model("ada/storage/checkpoints/ada_core_trained.pt")
    trainer.save_checkpoint("ada/storage/checkpoints/training_checkpoint.pt")
    
    print("✅ Quick test completed!")
    return history

if __name__ == "__main__":
    run_quick_test()