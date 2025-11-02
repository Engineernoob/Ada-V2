"""
🌍 Environment for Ada's Reinforcement Learning
Simulates conversational scenarios for training
"""

import random
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class ConversationState:
    """State representation for conversational environment"""
    user_input: str
    context: str
    conversation_history: List[str]
    turn_count: int
    emotional_state: str = "neutral"  # positive, negative, neutral
    
    def get_state_vector(self) -> List[float]:
        """Convert state to numerical representation for neural network"""
        vector = [
            len(self.user_input) / 100.0,  # Normalized input length
            self.turn_count / 50.0,  # Normalized turn count
            1.0 if self.emotional_state == "positive" else 0.0,
            1.0 if self.emotional_state == "negative" else 0.0,
        ]
        return vector

class AdaEnvironment:
    """Simulated environment for Ada's reinforcement learning"""
    
    def __init__(self):
        self.conversation_scenarios = [
            {"input": "Hello Ada", "expected_response": "greeting"},
            {"input": "How are you?", "expected_response": "emotional"},
            {"input": "Can you help me?", "expected_response": "helpful"},
            {"input": "Tell me a joke", "expected_response": "creative"},
            {"input": "What's the weather?", "expected_response": "factual"},
            {"input": "I'm sad", "expected_response": "empathetic"},
            {"input": "Thank you", "expected_response": "grateful"},
            {"input": "Goodbye", "expected_response": "farewell"},
        ]
        
        self.current_scenario_idx = 0
        self.turn_count = 0
        self.conversation_history = []
        
    def get_current_state(self, user_input: str = "") -> ConversationState:
        """Get current conversational state"""
        scenario = self.conversation_scenarios[self.current_scenario_idx]
        
        state = ConversationState(
            user_input=user_input or scenario["input"],
            context=scenario["expected_response"],
            conversation_history=self.conversation_history.copy(),
            turn_count=self.turn_count,
            emotional_state=self._detect_emotion(user_input)
        )
        
        return state
    
    def step(self, user_input: str, ada_response: str) -> Tuple[ConversationState, float, bool]:
        """Take a step in the environment"""
        self.turn_count += 1
        
        # Get current state
        state = self.get_current_state(user_input)
        
        # Add to conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Ada: {ada_response}")
        
        # Compute reward
        reward = self._compute_reward(user_input, ada_response, state.context)
        
        # Check if scenario is complete
        done = self._is_scenario_complete(ada_response, state.context)
        
        # Move to next scenario if current is complete
        if done:
            self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.conversation_scenarios)
            self.turn_count = 0
            self.conversation_history = []
        
        return state, reward, done
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_scenario_idx = 0
        self.turn_count = 0
        self.conversation_history = []
        print("🔄 Environment reset")
    
    def _detect_emotion(self, text: str) -> str:
        """Simple emotion detection from text"""
        text = text.lower()
        
        positive_words = ["happy", "good", "great", "excellent", "wonderful", "amazing", "love", "like"]
        negative_words = ["sad", "bad", "terrible", "awful", "hate", "angry", "upset", "frustrated"]
        
        if any(word in text for word in positive_words):
            return "positive"
        elif any(word in text for word in negative_words):
            return "negative"
        else:
            return "neutral"
    
    def _compute_reward(self, user_input: str, ada_response: str, expected_context: str) -> float:
        """Compute reward based on response quality"""
        reward = 0.0
        response_lower = ada_response.lower()
        
        # Base reward for any response
        reward += 0.1
        
        # Context-specific rewards
        if expected_context == "greeting":
            if any(word in response_lower for word in ["hello", "hi", "greetings", "nice", "meet"]):
                reward += 0.5
        elif expected_context == "emotional":
            if any(word in response_lower for word in ["feel", "doing", "good", "fine", "well"]):
                reward += 0.4
        elif expected_context == "helpful":
            if any(word in response_lower for word in ["help", "can", "assist", "sure", "absolutely"]):
                reward += 0.4
        elif expected_context == "empathetic":
            if any(word in response_lower for word in ["sorry", "understand", "feel", "that", "tough"]):
                reward += 0.4
        elif expected_context == "creative":
            if any(word in response_lower for word in ["joke", "funny", "laugh", "humor"]):
                reward += 0.4
        elif expected_context == "grateful":
            if any(word in response_lower for word in ["welcome", "glad", "happy", "helpful"]):
                reward += 0.4
        
        # Response length bonus
        response_words = len(response_lower.split())
        if 2 <= response_words <= 20:
            reward += 0.1
        elif response_words < 2:
            reward -= 0.1
        
        # Penalize empty or very short responses
        if len(ada_response.strip()) < 2:
            reward -= 0.3
        
        return max(0.0, min(1.0, reward))
    
    def _is_scenario_complete(self, ada_response: str, expected_context: str) -> bool:
        """Check if the current scenario is complete"""
        # Simple completion check based on response patterns
        response_lower = ada_response.lower()
        
        if expected_context == "greeting":
            return any(word in response_lower for word in ["hello", "hi", "greetings"])
        elif expected_context == "farewell":
            return any(word in response_lower for word in ["goodbye", "bye", "see", "later"])
        elif expected_context == "grateful":
            return any(word in response_lower for word in ["welcome", "glad", "happy"])
        else:
            # For other contexts, consider complete after one response
            return len(ada_response.strip()) > 0
    
    def get_scenario_count(self) -> int:
        """Get total number of available scenarios"""
        return len(self.conversation_scenarios)
    
    def get_current_scenario(self) -> Dict:
        """Get current scenario information"""
        return self.conversation_scenarios[self.current_scenario_idx]

if __name__ == "__main__":
    # Test the environment
    env = AdaEnvironment()
    
    print("🧪 Testing Ada Environment...")
    print(f"Total scenarios: {env.get_scenario_count()}")
    
    # Test a conversation flow
    for i in range(5):
        state = env.get_current_state()
        print(f"\nScenario {i+1}:")
        print(f"User input: {state.user_input}")
        print(f"Expected context: {state.context}")
        
        # Simulate Ada's response
        mock_response = f"This is my response to '{state.user_input}'"
        
        # Step through environment
        new_state, reward, done = env.step(state.user_input, mock_response)
        print(f"Ada response: {mock_response}")
        print(f"Reward: {reward:.3f}")
        print(f"Scenario complete: {done}")