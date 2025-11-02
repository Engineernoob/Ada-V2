"""
🔤 Text Encoder for Ada
Converts text input into embeddings for neural processing
"""

import torch
import numpy as np
from typing import List, Dict
import re

class TextEncoder:
    """Simple text encoder using character-level encoding"""
    
    def __init__(self, max_length=512):
        self.max_length = max_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self._build_vocab()
    
    def _build_vocab(self):
        """Build character-level vocabulary"""
        # Common characters for English text
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,:;!?()[]{}'\"-_\n\t"
        
        self.char_to_idx = {char: idx for idx, char in enumerate(chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(chars)
        print(f"📚 Built vocabulary with {self.vocab_size} characters")
    
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to tensor of indices"""
        # Clean and truncate text
        text = text.strip()[:self.max_length]
        
        # Convert to indices
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[' '])  # Default to space for unknown chars
        
        # Pad to max_length
        while len(indices) < self.max_length:
            indices.append(0)  # Padding token
        
        return torch.tensor(indices[:self.max_length], dtype=torch.long)
    
    def decode(self, tensor: torch.Tensor) -> str:
        """Convert tensor of indices back to text"""
        indices = tensor.tolist()
        text = ""
        for idx in indices:
            if idx in self.idx_to_char:
                text += self.idx_to_char[idx]
            else:
                text += ' '  # Default to space for unknown indices
        
        return text.strip()
    
    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts"""
        encoded_batch = []
        for text in texts:
            encoded = self.encode(text)
            encoded_batch.append(encoded)
        return torch.stack(encoded_batch)
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension for the neural network"""
        return self.vocab_size

def simple_text_embedding(text: str, embedding_dim=512) -> torch.Tensor:
    """Generate a simple embedding for text using bag-of-characters approach"""
    # Simple character frequency embedding
    text = text.lower()
    char_counts = {}
    
    # Count character frequencies
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # Normalize to create embedding
    total_chars = len(text) if len(text) > 0 else 1
    embedding = torch.zeros(embedding_dim)
    
    for i, char in enumerate('abcdefghijklmnopqrstuvwxyz'):
        if char in char_counts:
            embedding[i] = char_counts[char] / total_chars
    
    # Add text length as feature
    embedding[-1] = min(len(text) / 1000.0, 1.0)  # Normalized length
    
    return embedding

if __name__ == "__main__":
    # Test the text encoder
    encoder = TextEncoder(max_length=128)
    
    test_text = "Hello Ada! How are you today?"
    encoded = encoder.encode(test_text)
    decoded = encoder.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded: {decoded}")
    print(f"Vocab size: {encoder.vocab_size}")
    
    # Test simple embedding
    simple_emb = simple_text_embedding(test_text)
    print(f"Simple embedding shape: {simple_emb.shape}")