"""
🎯 Reward Model for Ada's Reinforcement Learning
Evaluates response quality for training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict

class RewardModel(nn.Module):
    """Simple reward model to evaluate response quality"""
    
    def __init__(self, input_dim=512, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single reward output
        self.relu = nn.ReLU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        """Forward pass to compute reward"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        reward = torch.sigmoid(self.fc3(x))  # Reward between 0 and 1
        return reward

class RuleBasedReward:
    """Simple rule-based reward system for Phase 1"""
    
    def __init__(self):
        self.positive_indicators = [
            "hello", "hi", "good", "great", "excellent", "wonderful", "amazing",
            "thank", "thanks", "please", "help", "yes", "sure", "absolutely"
        ]
        
        self.negative_indicators = [
            "bad", "terrible", "awful", "stupid", "hate", "wrong", "error",
            "fail", "broken", "worst", "horrible", "disappointing"
        ]
        
        self.response_starters = [
            "hello", "hi", "greetings", "good", "nice", "happy"
        ]
    
    def compute_reward(self, user_input: str, ada_response: str) -> float:
        """Compute reward based on simple heuristics"""
        reward = 0.5  # Base reward
        
        user_lower = user_input.lower()
        response_lower = ada_response.lower()
        
        # Check for positive indicators in response
        for indicator in self.positive_indicators:
            if indicator in response_lower:
                reward += 0.1
        
        # Check for negative indicators in response
        for indicator in self.negative_indicators:
            if indicator in response_lower:
                reward -= 0.1
        
        # Check for proper greeting patterns
        if any(greeting in user_lower for greeting in ["hello", "hi", "hey"]):
            if any(greeting in response_lower for greeting in ["hello", "hi", "greetings"]):
                reward += 0.2
        
        # Response length penalty/bonus
        response_words = len(response_lower.split())
        if 3 <= response_words <= 20:  # Sweet spot for responses
            reward += 0.1
        elif response_words < 2:  # Too short
            reward -= 0.1
        elif response_words > 50:  # Too long
            reward -= 0.05
        
        # Ensure reward is in valid range
        return max(0.0, min(1.0, reward))
    
    def get_reward_summary(self, user_input: str, ada_response: str) -> Dict:
        """Get detailed reward breakdown"""
        reward = self.compute_reward(user_input, ada_response)
        
        user_lower = user_input.lower()
        response_lower = ada_response.lower()
        
        matched_positive = [ind for ind in self.positive_indicators if ind in response_lower]
        matched_negative = [ind for ind in self.negative_indicators if ind in response_lower]
        
        return {
            "total_reward": reward,
            "positive_markers": matched_positive,
            "negative_markers": matched_negative,
            "response_length": len(ada_response.split()),
            "greeting_bonus": any(g in user_lower for g in ["hello", "hi", "hey"]) and
                             any(g in response_lower for g in ["hello", "hi", "greetings"])
        }

if __name__ == "__main__":
    # Test the reward model
    reward_system = RuleBasedReward()
    
    # Test cases
    test_cases = [
        ("Hello Ada", "Hello! I'm Ada, nice to meet you!"),
        ("Hello", "Hi there!"),
        ("How are you?", "I'm doing great, thank you!"),
        ("Tell me something bad", "I can't help with that."),
    ]
    
    print("🧪 Testing Reward Model...")
    for user_input, ada_response in test_cases:
        reward = reward_system.compute_reward(user_input, ada_response)
        summary = reward_system.get_reward_summary(user_input, ada_response)
        
        print(f"\nInput: {user_input}")
        print(f"Response: {ada_response}")
        print(f"Reward: {reward:.3f}")
        print(f"Summary: {summary}")