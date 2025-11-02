"""
🧠 Enhanced Neural Core for Ada v2.0
Integrates persona system, context memory, and reinforcement learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import json
from pathlib import Path

# Import Phase 2 components (with fallbacks)
try:
    from core.config import *
    from core.persona import get_current_persona, get_available_personas
    from core.memory import get_conversation_context
    from rl.reward_engine import reward_engine, get_reward_analysis
except ImportError:
    # Fallback values for standalone testing
    BASE_MODEL = "microsoft/DialoGPT-medium"
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MEMORY_LIMIT = 6
    CONTEXT_WINDOW = 6
    LR = 0.001
    FALLBACK_RESPONSE = "I apologize, but I'm having trouble processing that. Could you try rephrasing?"

# Silence Hugging Face warnings
import transformers
transformers.utils.logging.set_verbosity_error()

# 🧩 AdaNet — trainable neural head (for reinforcement tuning)
class AdaNet(nn.Module):
    def __init__(self, input_size=512, hidden=256, output_size=512):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 🧠 Enhanced AdaCore — integrates all Phase 2 components
class AdaCore:
    def __init__(self):
        print(f"🧠 Loading Ada v2.0 with Phase 2 enhancements...")
        
        # Load base language model
        print(f"📚 Loading base model: {BASE_MODEL}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.lm = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("📝 Using fallback mode...")
            self.lm = None
            self.tokenizer = None
        
        # Initialize AdaNet for reinforcement learning
        self.head = AdaNet()
        self.optimizer = optim.Adam(self.head.parameters(), lr=LR)
        self.rewards = []
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.lm:
            self.lm.to(self.device)
        self.head.to(self.device)
        
        print(f"✅ AdaCore v2.0 initialized on {self.device.upper()}")
        print(f"🎭 Persona system: Ready")
        print(f"🧠 Memory system: Ready")
        print(f"🎯 Reward system: Ready")
    
    # Enhanced conversational inference with context and persona
    def infer_with_context(self, messages: List[Dict], generation_params: Dict = None) -> str:
        """Enhanced inference with conversation context and persona integration"""
        
        try:
            # If model not loaded, return simple fallback
            if self.lm is None or self.tokenizer is None:
                return self._simple_response(messages)
            
            # Get current persona for system instructions
            try:
                persona = get_current_persona()
                system_prompt = persona.get_system_prompt() if persona else "You are Ada, a helpful AI assistant."
            except:
                system_prompt = "You are Ada, a helpful AI assistant."
            
            # Prepare the full conversation context
            full_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation context if available
            try:
                context = get_conversation_context(CONTEXT_WINDOW)
                if context:
                    full_messages.append({"role": "system", "content": f"Previous conversation context:\n{context}"})
            except:
                pass  # Continue without context
            
            # Add the current messages
            full_messages.extend(messages)
            
            # Apply persona-specific generation parameters
            if generation_params is None:
                generation_params = {"temperature": TEMPERATURE, "top_p": TOP_P}
            
            # Use persona temperature and top_p
            temperature = generation_params.get("temperature", TEMPERATURE)
            top_p = generation_params.get("top_p", TOP_P)
            
            # Prepare the prompt for the model
            text_prompt = self.tokenizer.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                text_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.lm.generate(
                    **inputs,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode and clean the response
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt part and clean response
            response = self._clean_response(text, len(text_prompt))
            
            return response
            
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return self._simple_response(messages)
    
    def infer(self, prompt: str) -> str:
        """Legacy inference method for compatibility"""
        messages = [{"role": "user", "content": prompt}]
        return self.infer_with_context(messages)
    
    def _simple_response(self, messages: List[Dict]) -> str:
        """Simple response generation for fallback mode"""
        if not messages:
            return FALLBACK_RESPONSE
        
        last_message = messages[-1].get("content", "")
        user_input = last_message.lower()
        
        # Simple response patterns
        if any(word in user_input for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Ada, your personal AI assistant. How can I help you today?"
        elif any(word in user_input for word in ["thank", "thanks"]):
            return "You're very welcome! I'm happy to help."
        elif "help" in user_input:
            return "I'm here to assist you! What would you like to explore?"
        elif "how are you" in user_input:
            return "I'm doing wonderfully, thank you for asking! How are you doing?"
        elif "?" in user_input:
            return "That's a great question! Let me think about that for a moment."
        else:
            return "That's interesting! Tell me more about that."
    
    def _clean_response(self, text: str, prompt_length: int) -> str:
        """Clean response by removing reasoning and system content"""
        
        # Extract just the assistant response part
        if "assistant" in text.lower():
            # Split at the last "assistant" occurrence
            parts = text.split("assistant")[-1]
            response = parts.strip()
        else:
            # Fallback: take everything after the prompt
            response = text[prompt_length:].strip()
        
        # Remove reasoning markers and internal thoughts
        reasoning_markers = [
            "<|im_start|>", "<|im_end|>", "system", "user", "assistant",
            "", "<|im_start|system|>", "<|im_start|user|>",
            "I need to respond", "Let me think", "I should", "So the user",
            "The user asked", "To respond", "my response should be", "I will",
            "Okay,", "Firstly,", "I notice", "I observe"
        ]
        
        for marker in reasoning_markers:
            response = response.replace(marker, "").strip()
        
        # Clean up formatting
        response = " ".join(response.split())  # Remove extra whitespace
        
        return response
    
    # Enhanced reinforcement learning integration
    def reinforce(self, reward: float):
        """Enhanced reinforcement with Phase 2 integration"""
        try:
            # Log explicit reward if reward engine available
            try:
                reward_engine.log_explicit_reward("", "", reward)
            except:
                pass  # Continue without reward logging
            
            self.rewards.append(reward)
            
            # Convert reward to tensor
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device)
            
            # Get prediction from AdaNet head
            pred = torch.mean(torch.stack([p.mean() for p in self.head.parameters()]))
            
            # Compute loss (negative because we want to maximize reward)
            loss = -reward_tensor * pred
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Display reinforcement result
            emoji = "🌟" if reward > 0 else ("⚖️" if reward == 0 else "🚫")
            print(f"{emoji} Reinforced with reward {reward:+.2f} | Loss {loss.item():.4f}")
            
        except Exception as e:
            print(f"⚠️ Reinforcement error: {e}")
    
    def save_brain(self, path="storage/models/adacore_head.pth"):
        """Save trained Ada neural head"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.head.state_dict(), path)
            print(f"💾 Ada neural head saved to {path}")
        except Exception as e:
            print(f"⚠️ Save error: {e}")
    
    def load_brain(self, path="storage/models/adacore_head.pth"):
        """Load trained Ada neural head"""
        try:
            if Path(path).exists():
                self.head.load_state_dict(torch.load(path, map_location=self.device))
                print("🧠 Loaded trained Ada neural head.")
                return True
            else:
                print("⚠️ No trained Ada head found; starting fresh.")
                return False
        except Exception as e:
            print(f"⚠️ Load error: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get AdaCore statistics"""
        return {
            "device": self.device,
            "model_loaded": self.lm is not None,
            "total_rewards": len(self.rewards),
            "average_reward": sum(self.rewards) / len(self.rewards) if self.rewards else 0.0,
            "personas_available": get_available_personas() if hasattr(self, 'get_available_personas') else ["friendly"]
        }

if __name__ == "__main__":
    # Test the AdaCore
    print("🧪 Testing AdaCore v2.0...")
    
    ada_core = AdaCore()
    
    # Test simple inference
    response = ada_core.infer("Hello Ada!")
    print(f"Response: {response}")
    
    # Test reinforcement
    ada_core.reinforce(0.8)
    
    # Test stats
    stats = ada_core.get_stats()
    print(f"Stats: {stats}")
    
    print("✅ AdaCore v2.0 test completed!")