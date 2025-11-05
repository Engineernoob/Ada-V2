"""
🧠 Enhanced Neural Core for Ada v3.0
Integrates long-term memory, reflection, and all Phase 2+3 components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.optim as optim
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime
from pathlib import Path

# Import all Phase 2+3 components (with fallbacks)
try:
    from core.config import *
    from core.persona import get_current_persona, get_available_personas
    from core.memory import get_conversation_context, add_to_memory, get_memory_session
    from rl.reward_engine import reward_engine, get_reward_analysis
    from core.long_memory import initialize_long_memory, add_to_long_memory, query_long_memory, LongMemory
    from core.reflection import initialize_reflection_system, reflect, ReflectionManager
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    # Fallback values for standalone testing
    BASE_MODEL = "microsoft/DialoGPT-medium"
    MAX_TOKENS = 150
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MEMORY_LIMIT = 6
    CONTEXT_WINDOW = 6
    LR = 0.001
    FALLBACK_RESPONSE = "I apologize, but I'm having trouble processing that. Could you try rephrasing?"
    
    # Phase-3 defaults
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    REFLECTION_INTERVAL = 10
    MEMORY_DB_PATH = "storage/memory/long_memory.faiss"

# Silence Hugging Face warnings
import transformers
transformers.utils.logging.set_verbosity_error()

# 🧩 AdaNet v3 — Adaptive policy head (context-aware reinforcement learning)
class AdaNet(nn.Module):
    """
    AdaNet v3 — Adaptive policy head.
    Learns tone, style, and reward prediction from text embeddings.
    """
    def __init__(self, input_size=768, hidden1=512, hidden2=256, output_size=3, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm(F.relu(self.fc2(x)))
        return torch.tanh(self.fc3(x))   # [style, tone, reward_pred]

# 🧠 Enhanced AdaCore v3.0 — integrates all Phase 2+3 components
class AdaCore:
    def __init__(self):
        print(f"🧠 Loading Ada v3.0 with Phase 3 enhancements...")
        
        # Enhanced device detection for Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🍎 Using Apple Silicon (MPS) GPU acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("🚀 Using CUDA GPU acceleration")
        else:
            self.device = torch.device("cpu")
            print("💻 Using CPU")
        
        # Load base language model
        print(f"📚 Loading base model: {BASE_MODEL}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model with proper device placement
            model_dtype = torch.float16 if self.device.type in ["mps", "cuda"] else torch.float32
            self.lm = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=model_dtype
            )
            self.lm.to(self.device)
            self.lm.eval()  # Set to evaluation mode
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print("📝 Using fallback mode...")
            self.lm = None
            self.tokenizer = None
        
        # Initialize AdaNet v3 for reinforcement learning with semantic embeddings
        from sentence_transformers import SentenceTransformer
        from rl.reward_memory import RewardMemory
        
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.head = AdaNet(input_size=384)  # Match sentence-transformers dimension
        self.head.to(self.device)
        self.optimizer = optim.AdamW(self.head.parameters(), lr=1e-4, betas=(0.9,0.99))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        self.rewards = []
        self.training_step = 0
        
        # Initialize RewardMemory buffer for AdaNet v3
        self.reward_memory = RewardMemory(capacity=500)
        print("🎯 RewardMemory buffer: Initialized")
        
        # Phase 3: Initialize long-term memory system
        try:
            self.long_memory = initialize_long_memory("storage/memory")
            print("🧠 Long-term memory: Initialized")
        except Exception as e:
            print(f"⚠️ Long-term memory error: {e}")
            self.long_memory = None
        
        # Phase 3: Initialize reflection system
        try:
            self.reflection_manager = initialize_reflection_system("storage/memory/summaries")
            print("🔮 Reflection system: Initialized")
        except Exception as e:
            print(f"⚠️ Reflection system error: {e}")
            self.reflection_manager = None
        
        # Session tracking for Phase 3
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.turn_count = 0
        self.session_history = []
        
        print(f"✅ AdaCore v3.0 initialized on {self.device.type.upper()}")
        print(f"🎭 Persona system: Ready")
        print(f"🧠 Memory system: Ready")
        print(f"🎯 Reward system: Ready")
        print(f"🔮 Long-term memory: {'✅' if self.long_memory else '❌'}")
        print(f"🔮 Reflection system: {'✅' if self.reflection_manager else '❌'}")
        print(f"📅 Session ID: {self.session_id}")
    
    # Enhanced conversational inference with Phase 3 integrations
    def infer_with_context(self, messages: List[Dict], generation_params: Dict = None) -> str:
        """Enhanced inference with conversation context and Phase 3 integrations"""
        
        try:
            # Extract current user input for Phase 3 processing
            current_user_input = ""
            if messages:
                last_message = messages[-1]
                if last_message.get("role") == "user":
                    current_user_input = last_message.get("content", "")
            
            # If model not loaded, return simple fallback
            if self.lm is None or self.tokenizer is None:
                return self._simple_response(messages)
            
            # Get current persona for system instructions
            try:
                persona = get_current_persona()
                system_prompt = persona.get_system_prompt() if persona else "You are Ada, a helpful AI assistant."
            except:
                system_prompt = "You are Ada, a helpful AI assistant."
            
            # Phase 3: Query long-term memory for relevant context
            long_term_context = ""
            if self.long_memory and current_user_input:
                try:
                    relevant_memories = query_long_memory(current_user_input, top_k=3)
                    if relevant_memories:
                        long_term_context = f"\nRelevant past context:\n" + "\n".join(
                            f"- {memory}" for memory in relevant_memories[:2]
                        )
                except Exception as e:
                    print(f"⚠️ Long-term memory query error: {e}")
            
            # Prepare the full conversation context
            full_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation context if available
            try:
                context = get_conversation_context(CONTEXT_WINDOW)
                if context:
                    full_messages.append({"role": "system", "content": f"Recent conversation:\n{context}"})
            except:
                pass  # Continue without context
            
            # Add long-term memory context
            if long_term_context:
                full_messages.append({"role": "system", "content": long_term_context})
            
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
            
            # Tokenize and move inputs to the correct device
            inputs = self.tokenizer(
                text_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=1024,
                padding=True
            )
            # Move all tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
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
            
            # Phase 3: Store interaction in long-term memory
            if self.long_memory and current_user_input and response:
                try:
                    memory_id = add_to_long_memory(
                        f"User: {current_user_input} | Ada: {response}", 
                        self.session_id, 
                        self.turn_count + 1
                    )
                    print(f"💾 Stored in long-term memory: {memory_id[:8]}...")
                except Exception as e:
                    print(f"⚠️ Long-term memory storage error: {e}")
            
            # Phase 3: Track session history for reflection
            self.turn_count += 1
            self.session_history.append({
                "user_input": current_user_input,
                "ada_response": response,
                "timestamp": datetime.now().isoformat(),
                "turn_number": self.turn_count
            })
            
            # Phase 3: Check if reflection is needed
            if self.turn_count % REFLECTION_INTERVAL == 0:
                self._periodic_reflection()
            
            return response
            
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            return self._simple_response(messages)
    
    def infer(self, prompt: str) -> str:
        """Legacy inference method for compatibility"""
        messages = [{"role": "user", "content": prompt}]
        return self.infer_with_context(messages)
    
    def _periodic_reflection(self):
        """Generate periodic reflections during long sessions"""
        if self.reflection_manager and len(self.session_history) >= REFLECTION_INTERVAL:
            try:
                print(f"🔮 Generating periodic reflection (turn {self.turn_count})...")
                avg_reward = sum(self.session_history[-REFLECTION_INTERVAL:][i].get('reward', 0) 
                               for i in range(REFLECTION_INTERVAL)) / REFLECTION_INTERVAL
                
                reflection_text = reflect(
                    self.session_history[-REFLECTION_INTERVAL:],
                    avg_reward=avg_reward,
                    session_id=f"{self.session_id}_periodic_{self.turn_count}"
                )
                
                print(f"📝 Periodic reflection completed")
                
            except Exception as e:
                print(f"⚠️ Periodic reflection error: {e}")
    
    def _simple_response(self, messages: List[Dict]) -> str:
        """Simple response generation for fallback mode"""
        if not messages:
            return FALLBACK_RESPONSE
        
        last_message = messages[-1].get("content", "")
        user_input = last_message.lower()
        
        # Simple response patterns
        if any(word in user_input for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Ada, your personal AI assistant with long-term memory. How can I help you today?"
        elif any(word in user_input for word in ["thank", "thanks"]):
            return "You're very welcome! I'm happy to help."
        elif "help" in user_input:
            return "I'm here to assist you! I can remember our conversations across sessions now. What would you like to explore?"
        elif "how are you" in user_input:
            return "I'm doing wonderfully! I'm excited about my new long-term memory and reflection capabilities."
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
            "", "I need to respond", "Let me think", "I should", 
            "So the user", "The user asked", "To respond", "my response should be", "I will",
            "Okay,", "Firstly,", "I notice", "I observe"
        ]
        
        for marker in reasoning_markers:
            response = response.replace(marker, "").strip()
        
        # Clean up formatting
        response = " ".join(response.split())  # Remove extra whitespace
        
        return response
    
    # AdaNet v3: Multi-objective reinforcement learning with entropy regularization
    def reinforce(self, prompt: str, response: str, reward: float):
        """
        Enhanced reinforcement with AdaNet v3 semantic learning.
        Stores experiences in RewardMemory and trains AdaNet v3.
        """
        try:
            # Store experience in RewardMemory buffer
            self.reward_memory.add(prompt, response, reward)
            
            # Log explicit reward if reward engine available
            try:
                reward_engine.log_explicit_reward(prompt, response, reward)
            except:
                pass  # Continue without reward logging
            
            self.rewards.append(reward)
            self.training_step += 1
            
            # Generate semantic embedding of the response for AdaNet v3 training
            embedding = self.embedder.encode([response], convert_to_tensor=True).to(self.device)
            # Detach and clone to ensure proper gradient computation
            embedding = embedding.detach().clone().requires_grad_(True)
            
            # Get AdaNet v3 output: [style, tone, reward_pred]
            style, tone, reward_pred = self.head(embedding)[0]
            
            # Multi-objective loss calculation
            r = torch.tensor(reward, dtype=torch.float32, device=self.device)
            
            policy_loss = F.mse_loss(reward_pred, r)
            entropy_loss = -0.01 * (torch.abs(style) + torch.abs(tone)).mean()
            loss = policy_loss + entropy_loss
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.optimizer.step()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Display reinforcement result
            emoji = "🌟" if reward > 0 else ("⚖️" if reward == 0 else "🚫")
            print(f"{emoji} AdaNet v3 Reinforcement | Policy {policy_loss.item():.4f} | Entropy {entropy_loss.item():.4f}")
            
            # Periodic replay training
            if self.training_step % 50 == 0:
                self.replay_train()
            
        except Exception as e:
            print(f"⚠️ AdaNet v3 Reinforcement error: {e}")
    
    def replay_train(self):
        """
        Replay training on past experiences to reinforce long-term trends.
        Samples experiences from RewardMemory and trains AdaNet v3.
        """
        try:
            if self.reward_memory.size() < 4:  # Need minimum experiences for training
                return
            
            print("🔄 AdaNet v3 Replay Training...")
            total_loss = 0.0
            batch_size = 8
            
            # Sample experiences for replay training
            experiences = self.reward_memory.sample(batch_size)
            
            for prompt, response, reward in experiences:
                # Encode response text for semantic learning
                embedding = self.embedder.encode([response], convert_to_tensor=True).to(self.device)
                # Detach and clone to ensure proper gradient computation
                embedding = embedding.detach().clone().requires_grad_(True)
                
                # Get AdaNet v3 prediction
                style, tone, reward_pred = self.head(embedding)[0]
                
                # Compute loss
                r = torch.tensor(reward, dtype=torch.float32, device=self.device)
                loss = F.mse_loss(reward_pred, r)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Update scheduler
            self.scheduler.step()
            
            avg_loss = total_loss / len(experiences)
            print(f"🎯 Replay Training Complete | Avg Loss: {avg_loss:.4f} | Experiences: {len(experiences)}")
            
        except Exception as e:
            print(f"⚠️ Replay training error: {e}")
    
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
    
    # Phase 3: Session management
    def save_session(self):
        """Save current session data"""
        try:
            session_file = f"storage/memory/session_{datetime.now().strftime('%Y-%m-%d_%H%M')}.jsonl"
            Path(session_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(session_file, 'w', encoding='utf-8') as f:
                for turn in self.session_history:
                    f.write(json.dumps(turn, ensure_ascii=False) + '\n')
            
            print(f"💾 Session saved: {session_file}")
            
        except Exception as e:
            print(f"⚠️ Session save error: {e}")
    
    def generate_final_reflection(self):
        """Generate final reflection at session end"""
        if self.reflection_manager and self.session_history:
            try:
                print("🔮 Generating final session reflection...")
                
                avg_reward = sum(turn.get('reward', 0) for turn in self.session_history) / len(self.session_history)
                
                reflection_text = reflect(
                    self.session_history,
                    avg_reward=avg_reward,
                    session_id=self.session_id
                )
                
                print("📝 Final reflection generated and saved")
                return reflection_text
                
            except Exception as e:
                print(f"⚠️ Final reflection error: {e}")
    
    def get_stats(self) -> Dict:
        """Get AdaCore statistics"""
        stats = {
            "device": self.device,
            "model_loaded": self.lm is not None,
            "total_rewards": len(self.rewards),
            "average_reward": sum(self.rewards) / len(self.rewards) if self.rewards else 0.0,
            "personas_available": get_available_personas() if hasattr(self, 'get_available_personas') else ["friendly"],
            "session_id": self.session_id,
            "turn_count": self.turn_count
        }
        
        # Add Phase 3 statistics
        if self.long_memory:
            try:
                stats["long_memory_stats"] = self.long_memory.get_statistics()
            except:
                stats["long_memory_stats"] = {"error": "Could not retrieve"}
        
        if self.reflection_manager:
            try:
                recent_reflections = self.reflection_manager.get_recent_reflections(5)
                stats["recent_reflections"] = len(recent_reflections)
            except:
                stats["recent_reflections"] = {"error": "Could not retrieve"}
        
        return stats

if __name__ == "__main__":
    # Test the AdaCore v3.0
    print("🧪 Testing AdaCore v3.0...")
    
    ada_core = AdaCore()
    
    # Test simple inference
    response = ada_core.infer("Hello Ada with long-term memory!")
    print(f"Response: {response}")
    
    # Test reinforcement
    ada_core.reinforce(0.8)
    
    # Test stats
    stats = ada_core.get_stats()
    print(f"Stats: {stats}")
    
    print("✅ AdaCore v3.0 test completed!")