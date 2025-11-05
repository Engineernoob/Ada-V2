"""
🎯 Reward Memory Buffer for AdaNet v3
Stores interaction experiences for replay training and learning stability
"""

import random
from typing import List, Tuple, Optional

class RewardMemory:
    """
    Reward Memory Buffer - stores interaction experiences for replay training.
    Maintains fixed-capacity buffer of (prompt, response, reward) tuples.
    """
    
    def __init__(self, capacity: int = 500):
        self.buffer: List[Tuple[str, str, float]] = []
        self.capacity = capacity
    
    def add(self, prompt: str, response: str, reward: float):
        """
        Add new experience to memory buffer.
        Maintains capacity by removing oldest entries when full.
        """
        experience = (prompt, response, reward)
        
        if len(self.buffer) >= self.capacity:
            # Remove oldest entry to maintain capacity
            self.buffer.pop(0)
        
        self.buffer.append(experience)
    
    def sample(self, n: int = 8) -> List[Tuple[str, str, float]]:
        """
        Sample n random experiences from memory buffer.
        Returns up to n experiences, or all if buffer has fewer than n.
        """
        if not self.buffer:
            return []
        
        sample_size = min(n, len(self.buffer))
        return random.sample(self.buffer, sample_size)
    
    def get_all(self) -> List[Tuple[str, str, float]]:
        """Get all stored experiences."""
        return self.buffer.copy()
    
    def clear(self):
        """Clear all experiences from memory."""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current number of stored experiences."""
        return len(self.buffer)
    
    def get_recent(self, n: int = 10) -> List[Tuple[str, str, float]]:
        """Get the n most recent experiences."""
        return self.buffer[-n:] if self.buffer else []
    
    def get_average_reward(self, recent_n: Optional[int] = None) -> float:
        """Calculate average reward of experiences."""
        if not self.buffer:
            return 0.0
        
        experiences = self.get_recent(recent_n) if recent_n else self.buffer
        rewards = [exp[2] for exp in experiences]  # Extract rewards
        
        return sum(rewards) / len(rewards)

if __name__ == "__main__":
    # Test the RewardMemory system
    print("🧪 Testing RewardMemory Buffer...")
    
    # Create buffer with capacity of 5
    memory = RewardMemory(capacity=5)
    
    # Add some test experiences
    test_experiences = [
        ("Hello Ada", "Hello! How can I help you today?", 0.8),
        ("What is Python?", "Python is a programming language...", 0.6),
        ("Tell me a joke", "Why did the chicken cross the road? To get to the other side!", 0.9),
        ("Help me with coding", "I'd be happy to help with programming!", 0.7),
        ("What's 2+2?", "2+2 equals 4", 0.5),
        ("Goodbye", "See you later!", 0.8)
    ]
    
    # Add experiences (should trigger capacity limit)
    for prompt, response, reward in test_experiences:
        memory.add(prompt, response, reward)
        print(f"Added: {prompt[:20]}... (reward: {reward})")
    
    print(f"\nMemory size: {memory.size()}")
    print(f"Average reward: {memory.get_average_reward():.3f}")
    
    # Test sampling
    print(f"\nSampled experiences:")
    for i, (prompt, response, reward) in enumerate(memory.sample(3), 1):
        print(f"{i}. {prompt[:30]}... -> {response[:30]}... (reward: {reward})")
    
    # Test recent experiences
    print(f"\nRecent experiences:")
    for i, (prompt, response, reward) in enumerate(memory.get_recent(3), 1):
        print(f"{i}. {prompt[:30]}... -> {response[:30]}... (reward: {reward})")
    
    print("\n✅ RewardMemory test completed!")