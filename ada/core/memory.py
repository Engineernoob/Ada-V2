"""
🧠 Context Memory System for Ada
Handles short-term recall, session summarization, and memory persistence
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

class ConversationTurn:
    """Represents a single conversation turn"""
    
    def __init__(self, user_input: str, ada_response: str, 
                 timestamp: str = None, turn_id: int = None):
        self.user_input = user_input
        self.ada_response = ada_response
        self.timestamp = timestamp or datetime.now().isoformat()
        self.turn_id = turn_id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "user_input": self.user_input,
            "ada_response": self.ada_response,
            "timestamp": self.timestamp,
            "turn_id": self.turn_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationTurn':
        """Create from dictionary"""
        return cls(
            user_input=data["user_input"],
            ada_response=data["ada_response"],
            timestamp=data.get("timestamp"),
            turn_id=data.get("turn_id")
        )

class SessionMemory:
    """Manages memory for a single conversation session"""
    
    def __init__(self, session_id: str, session_file_path: str):
        self.session_id = session_id
        self.session_file_path = session_file_path
        self.turns: List[ConversationTurn] = []
        self.short_term_memory: List[ConversationTurn] = []
        self.max_short_term_turns = 6  # Last 6 turns for context
        self.current_turn_id = 0
        self.load_session()
    
    def add_turn(self, user_input: str, ada_response: str) -> ConversationTurn:
        """Add a new conversation turn"""
        self.current_turn_id += 1
        turn = ConversationTurn(
            user_input=user_input,
            ada_response=ada_response,
            timestamp=datetime.now().isoformat(),
            turn_id=self.current_turn_id
        )
        
        self.turns.append(turn)
        self.short_term_memory.append(turn)
        
        # Maintain short-term memory size
        if len(self.short_term_memory) > self.max_short_term_turns:
            self.short_term_memory.pop(0)
        
        return turn
    
    def get_recent_turns(self, count: int = 6) -> List[ConversationTurn]:
        """Get the most recent conversation turns"""
        return self.short_term_memory[-count:] if self.short_term_memory else []
    
    def get_context_string(self, include_last_n: int = 6) -> str:
        """Get formatted context string from recent turns"""
        recent_turns = self.get_recent_turns(include_last_n)
        
        if not recent_turns:
            return ""
        
        context_lines = ["Previous conversation context:"]
        for turn in recent_turns:
            context_lines.append(f"User: {turn.user_input}")
            context_lines.append(f"Ada: {turn.ada_response}")
        
        return "\n".join(context_lines)
    
    def summarize_session(self) -> str:
        """Generate a summary of the current session"""
        if not self.turns:
            return "No conversation yet."
        
        # Simple summary based on turn count and topics
        user_inputs = [turn.user_input for turn in self.turns]
        first_user_input = user_inputs[0] if user_inputs else "Unknown"
        
        # Count turn types
        question_count = sum(1 for inp in user_inputs if "?" in inp)
        greeting_count = sum(1 for inp in user_inputs if any(greet in inp.lower() 
                          for greet in ["hello", "hi", "hey", "good morning", "good afternoon"]))
        
        summary_parts = []
        summary_parts.append(f"Session {self.session_id} summary:")
        summary_parts.append(f"- Total turns: {len(self.turns)}")
        
        if greeting_count > 0:
            summary_parts.append(f"- Greetings detected: {greeting_count}")
        
        if question_count > 0:
            summary_parts.append(f"- Questions asked: {question_count}")
        
        # First interaction
        summary_parts.append(f"- Started with: '{first_user_input[:50]}{'...' if len(first_user_input) > 50 else ''}'")
        
        return "\n".join(summary_parts)
    
    def save_session(self):
        """Save session to JSONL file"""
        try:
            session_data = {
                "session_id": self.session_id,
                "created_at": self.turns[0].timestamp if self.turns else datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_turns": len(self.turns),
                "turns": [turn.to_dict() for turn in self.turns]
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.session_file_path), exist_ok=True)
            
            # Append to JSONL file
            with open(self.session_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(session_data, ensure_ascii=False) + '\n')
        
        except Exception as e:
            print(f"⚠️ Error saving session: {e}")
    
    def load_session(self):
        """Load session from JSONL file"""
        if not os.path.exists(self.session_file_path):
            return
        
        try:
            with open(self.session_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        session_data = json.loads(line.strip())
                        if session_data.get("session_id") == self.session_id:
                            # Load existing turns
                            turns_data = session_data.get("turns", [])
                            self.turns = [ConversationTurn.from_dict(turn_data) for turn_data in turns_data]
                            self.current_turn_id = max((turn.turn_id for turn in self.turns), default=0)
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error loading session: {e}")

class MemoryManager:
    """Central memory management system for Ada"""
    
    def __init__(self, memory_dir: str = "storage/memory"):
        self.memory_dir = Path(memory_dir)
        self.sessions: Dict[str, SessionMemory] = {}
        self.current_session_id = None
        self.ensure_memory_directory()
    
    def ensure_memory_directory(self):
        """Ensure memory directory exists"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
    
    def start_new_session(self, session_id: str = None) -> str:
        """Start a new conversation session"""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        session_file = self.memory_dir / "session.jsonl"
        
        self.sessions[session_id] = SessionMemory(session_id, str(session_file))
        return session_id
    
    def get_current_session(self) -> Optional[SessionMemory]:
        """Get the current active session"""
        if self.current_session_id and self.current_session_id in self.sessions:
            return self.sessions[self.current_session_id]
        return None
    
    def add_to_memory(self, user_input: str, ada_response: str) -> ConversationTurn:
        """Add a turn to the current session"""
        session = self.get_current_session()
        if session is None:
            session = SessionMemory("default", str(self.memory_dir / "session.jsonl"))
            self.sessions["default"] = session
            self.current_session_id = "default"
        
        turn = session.add_turn(user_input, ada_response)
        session.save_session()
        return turn
    
    def get_context_for_inference(self, max_turns: int = 6) -> str:
        """Get formatted context for model inference"""
        session = self.get_current_session()
        if session:
            return session.get_context_string(max_turns)
        return ""
    
    def get_session_summary(self) -> str:
        """Get summary of current session"""
        session = self.get_current_session()
        if session:
            return session.summarize_session()
        return "No active session."
    
    def get_recent_topics(self, count: int = 5) -> List[str]:
        """Extract recent topics from conversation"""
        session = self.get_current_session()
        if not session:
            return []
        
        recent_turns = session.get_recent_turns(count)
        topics = []
        
        for turn in recent_turns:
            user_input = turn.user_input.lower()
            # Simple topic extraction
            if any(word in user_input for word in ["help", "problem", "issue", "question"]):
                topics.append("help-seeking")
            elif any(word in user_input for word in ["story", "creative", "imagine", "creative"]):
                topics.append("creative")
            elif any(word in user_input for word in ["code", "programming", "technical"]):
                topics.append("technical")
            elif any(word in user_input for word in ["weather", "news", "information"]):
                topics.append("information-seeking")
        
        return list(set(topics))  # Remove duplicates
    
    def cleanup_old_sessions(self, days_to_keep: int = 30):
        """Clean up sessions older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        # This is a simplified cleanup - in practice you'd want to track creation dates
        print(f"🧹 Memory cleanup scheduled for sessions older than {days_to_keep} days")

# Global memory manager instance
memory_manager = MemoryManager()

def start_memory_session(session_id: str = None) -> str:
    """Start a new memory session"""
    return memory_manager.start_new_session(session_id)

def add_to_memory(user_input: str, ada_response: str) -> ConversationTurn:
    """Add conversation turn to memory"""
    return memory_manager.add_to_memory(user_input, ada_response)

def get_memory_session():
    """Get the current active memory session (for compatibility)"""
    return memory_manager.get_current_session()

def get_conversation_context(max_turns: int = 6) -> str:
    """Get conversation context for inference"""
    return memory_manager.get_context_for_inference(max_turns)

if __name__ == "__main__":
    # Test the memory system
    print("🧪 Testing Memory System...")
    
    # Start a new session
    session_id = start_memory_session("test_session_001")
    print(f"Started session: {session_id}")
    
    # Add some conversation turns
    add_to_memory("Hello Ada, how are you?", "Hello! I'm doing great, thank you for asking!")
    add_to_memory("Can you help me with something?", "Of course! I'm here to help you with whatever you need.")
    add_to_memory("What's the weather like?", "I don't have access to real-time weather data, but I can help you check the weather through other means.")
    
    # Test context retrieval
    context = get_conversation_context(3)
    print(f"Context for inference:\n{context}")
    
    # Test session summary
    summary = memory_manager.get_session_summary()
    print(f"\nSession summary:\n{summary}")
    
    # Test recent topics
    topics = memory_manager.get_recent_topics()
    print(f"Recent topics: {topics}")
    
    print("✅ Memory system test completed!")