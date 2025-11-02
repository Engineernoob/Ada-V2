"""
💾 Memory Buffer for RL Experience Replay
Stores and manages experience data for training
"""

from typing import List, Dict, Any
import random
from collections import deque
import json

class MemoryBuffer:
    """Simple memory buffer for experience replay in RL"""
    
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.stats = {
            "total_experiences": 0,
            "average_reward": 0.0,
            "last_updated": None
        }
    
    def add_experience(self, experience: Dict[str, Any]):
        """Add an experience to the buffer"""
        self.buffer.append(experience)
        self.stats["total_experiences"] += 1
        self._update_stats()
    
    def add_batch(self, experiences: List[Dict[str, Any]]):
        """Add a batch of experiences"""
        for exp in experiences:
            self.add_experience(exp)
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a random batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)
    
    def get_recent_experiences(self, count: int) -> List[Dict[str, Any]]:
        """Get the most recent experiences"""
        return list(self.buffer)[-count:]
    
    def get_high_reward_experiences(self, min_reward: float, limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiences with rewards above threshold"""
        high_reward_exp = [
            exp for exp in self.buffer 
            if exp.get('reward', 0) >= min_reward
        ]
        return high_reward_exp[-limit:]  # Return most recent
    
    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer.clear()
        self.stats["total_experiences"] = 0
        self._update_stats()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.stats,
            "buffer_size": len(self.buffer),
            "max_capacity": self.max_size
        }
    
    def _update_stats(self):
        """Update internal statistics"""
        if len(self.buffer) > 0:
            rewards = [exp.get('reward', 0) for exp in self.buffer]
            self.stats["average_reward"] = sum(rewards) / len(rewards)
        else:
            self.stats["average_reward"] = 0.0
        
        from datetime import datetime
        self.stats["last_updated"] = datetime.now().isoformat()
    
    def save_to_file(self, filepath: str):
        """Save buffer to JSON file"""
        data = {
            "experiences": list(self.buffer),
            "statistics": self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Memory buffer saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load buffer from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.buffer = deque(data["experiences"], maxlen=self.max_size)
            self.stats = data.get("statistics", {})
            
            print(f"📂 Memory buffer loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"⚠️ Buffer file not found: {filepath}")
            return False
        except Exception as e:
            print(f"❌ Error loading buffer: {e}")
            return False

if __name__ == "__main__":
    # Test the memory buffer
    buffer = MemoryBuffer(max_size=100)
    
    # Add some dummy experiences
    for i in range(20):
        experience = {
            "state": [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            "action": [0.5 * i, 0.6 * i, 0.7 * i],
            "reward": 0.5 + 0.1 * i,
            "done": i % 5 == 4  # Done every 5 steps
        }
        buffer.add_experience(experience)
    
    # Test sampling
    batch = buffer.sample_batch(5)
    print(f"Sampled batch size: {len(batch)}")
    print(f"Buffer statistics: {buffer.get_statistics()}")
    
    # Test filtering
    high_reward = buffer.get_high_reward_experiences(0.7)
    print(f"High reward experiences: {len(high_reward)}")
    
    # Test save/load
    buffer.save_to_file("ada/storage/memory_buffer_test.json")
    
    new_buffer = MemoryBuffer()
    new_buffer.load_from_file("ada/storage/memory_buffer_test.json")
    print(f"Loaded buffer size: {len(new_buffer.buffer)}")