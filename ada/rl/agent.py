"""
🤖 RL Agent for Ada
Simple reinforcement learning agent for conversational training
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
import random

class AdaAgent:
    """Simple RL agent for Ada's conversational training"""
    
    def __init__(self, neural_network, learning_rate=0.001, epsilon=0.1):
        self.neural_network = neural_network
        self.device = neural_network._get_device()
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Exploration rate
        self.q_values = {}  # Simple Q-value storage
        self.experience_buffer = []
        
    def act(self, state_vector: List[float], training: bool = True) -> List[float]:
        """Choose action based on current state"""
        state_key = tuple(state_vector[:4])  # Use first 4 features as state key
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action = np.random.rand(10)  # 10-dimensional action space
            return action.tolist()
        else:
            # Exploitation: use neural network
            with torch.no_grad():
                state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                action = self.neural_network(state_tensor)
                return action.squeeze().cpu().numpy().tolist()
    
    def update(self, state: List[float], action: List[float], reward: float, 
               next_state: List[float], done: bool):
        """Update Q-values using simple Q-learning"""
        state_key = tuple(state[:4])
        next_state_key = tuple(next_state[:4])
        
        # Initialize Q-values if not seen before
        if state_key not in self.q_values:
            self.q_values[state_key] = np.random.rand(10)
        if next_state_key not in self.q_values:
            self.q_values[next_state_key] = np.random.rand(10)
        
        current_q = self.q_values[state_key]
        next_q = self.q_values[next_state_key]
        
        # Simple Q-learning update
        if done:
            target_q = reward
        else:
            target_q = reward + 0.95 * np.max(next_q)
        
        # Update the Q-value for the action actually taken
        action_idx = np.argmax(action)
        current_q[action_idx] += self.learning_rate * (target_q - current_q[action_idx])
        
        # Store experience for replay (simple buffer)
        self.experience_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        # Limit buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer.pop(0)
    
    def train_batch(self, batch_size=32):
        """Train on a batch of experiences"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.experience_buffer, batch_size)
        
        states = torch.tensor([exp['state'] for exp in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([exp['next_state'] for exp in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32).to(self.device)
        
        # Simple training logic
        with torch.enable_grad():
            current_q = self.neural_network(states)
            next_q = self.neural_network(next_states).detach()
            
            # Compute targets
            targets = rewards + (0.95 * torch.max(next_q, dim=1)[0] * (1 - dones))
            
            # Simple loss
            loss = torch.mean((current_q.mean(dim=1) - targets) ** 2)
            
            # Backward pass (simplified)
            loss.backward()
        
        return loss.item()
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            "experience_buffer_size": len(self.experience_buffer),
            "unique_states": len(self.q_values),
            "epsilon": self.epsilon
        }

if __name__ == "__main__":
    # Test the agent
    from ada.neural.policy_network import AdaCore
    
    ada_core = AdaCore()
    agent = AdaAgent(ada_core)
    
    # Test action selection
    state = [0.5, 0.1, 1.0, 0.0]  # Example state vector
    action = agent.act(state, training=False)
    print(f"State: {state}")
    print(f"Action: {action[:3]}...")  # Show first 3 elements
    
    # Test update
    agent.update(state, action, 0.8, [0.6, 0.2, 0.0, 1.0], False)
    
    print(f"Agent stats: {agent.get_stats()}")