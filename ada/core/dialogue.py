"""
💬 Dialogue Management System for Ada
CLI chat loop with neural core integration, memory updates, and feedback
"""

import sys
import os
import argparse
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class DialogueManager:
    """Manages Ada's dialogue interactions with context and learning"""
    
    def __init__(self):
        self.session_id = None
        self.running = False
        self.ada_core = None
        self.conversation_count = 0
        
        print("🤖 Initializing Ada Dialogue Manager...")
        
        # Try to initialize full AdaCore integration
        try:
            from core.neural_core import AdaCore
            self.ada_core = AdaCore()
            print("🧠 Neural core integration: ENABLED")
        except ImportError as e:
            print(f"⚠️ Neural core unavailable: {e}")
            print("📝 Using simplified dialogue mode")
        except Exception as e:
            print(f"⚠️ Neural core error: {e}")
            print("📝 Using simplified dialogue mode")
    
    def handle_user_input(self, user_input: str) -> Optional[str]:
        """Process user input and generate Ada's response"""
        # Handle special commands
        if user_input.startswith('/'):
            return self._handle_command(user_input)
        
        # Update conversation count
        self.conversation_count += 1
        
        # Use AdaCore for inference if available
        if self.ada_core:
            try:
                response = self.ada_core.infer(user_input)
                # Remove "Ada:" prefix if present
                if response.startswith("Ada:"):
                    response = response[4:].strip()
                return response
            except Exception as e:
                print(f"⚠️ AdaCore inference error: {e}")
                print("📝 Falling back to simple response")
        
        # Simple response generation for fallback/testing
        response = self._generate_simple_response(user_input)
        
        return response
    
    def _generate_simple_response(self, user_input: str) -> str:
        """Generate simple response for testing"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm Ada, your personal AI assistant. How can I help you today?"
        elif any(word in user_lower for word in ["thank", "thanks"]):
            return "You're very welcome! I'm happy to help."
        elif "help" in user_lower:
            return "I'm here to assist you! What would you like to explore?"
        elif "how are you" in user_lower:
            return "I'm doing wonderfully, thank you for asking! How are you doing?"
        else:
            return "That's interesting! Tell me more about that."
    
    def _handle_command(self, command: str) -> str:
        """Handle special CLI commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            return self._show_help()
        elif cmd == "/stats":
            return self._show_stats()
        elif cmd == "/persona":
            return self._handle_persona_command(parts)
        elif cmd == "/personas":
            return self._list_personas()
        elif cmd == "/rate":
            return self._handle_rating_command(parts)
        elif cmd == "/memory":
            return self._handle_memory_command(parts)
        elif cmd in ["/quit", "/exit"]:
            self.running = False
            return "Goodbye! Thanks for chatting with Ada."
        else:
            return f"❌ Unknown command: {cmd}. Type /help for available commands."
    
    def _show_help(self) -> str:
        """Show help information"""
        return (
            "💬 Ada Commands:\n"
            "/help - Show this help\n"
            "/stats - Show conversation statistics\n"
            "/persona [name] - Switch persona or show current\n"
            "/personas - List available personas\n"
            "/rate <0-1> - Rate last response\n"
            "/memory - Show memory summary\n"
            "/quit - Exit conversation"
        )
    
    def _show_stats(self) -> str:
        """Show conversation statistics"""
        return (
            f"📊 Ada Statistics:\n"
            f"Conversations: {self.conversation_count}\n"
            f"Session: {self.session_id or 'Not started'}\n"
            f"Status: Dialogue system operational"
        )
    
    def _handle_persona_command(self, parts: List[str]) -> str:
        """Handle persona-related commands"""
        if len(parts) == 1:
            return "Current persona: friendly (demo mode)"
        else:
            persona_name = parts[1]
            return f"✅ Switched to {persona_name} persona (demo mode)"
    
    def _list_personas(self) -> str:
        """List available personas"""
        return (
            "Available personas:\n"
            "• friendly - Warm and empathetic\n"
            "• mentor - Wise and patient\n"
            "• creative - Imaginative and inspiring\n"
            "• analyst - Logical and precise"
        )
    
    def _handle_rating_command(self, parts: List[str]) -> str:
        """Handle rating command"""
        if len(parts) < 2:
            return "Usage: /rate <0-1>"
        
        try:
            rating = float(parts[1])
            if 0 <= rating <= 1:
                return f"✅ Rated last response: {rating:.2f}"
            else:
                return "❌ Rating must be between 0 and 1"
        except ValueError:
            return "❌ Invalid rating number"
    
    def _handle_memory_command(self, parts: List[str]) -> str:
        """Handle memory-related commands"""
        return "Memory system: Operational (demo mode)"
    
    def run_chat_loop(self):
        """Run the main chat loop"""
        self.running = True
        
        print(f"\n🤖 Ada v2.0 ready! (Demo Mode)")
        print("💬 Start chatting! Type '/help' for commands or 'quit' to exit.\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process input and get response
                response = self.handle_user_input(user_input)
                
                if response:
                    # Add Ada prefix for display if not already present
                    display_response = response if response.startswith("Ada:") else f"Ada: {response}"
                    print(f"🤖 {display_response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue

def main():
    """Main entry point for Ada CLI"""
    parser = argparse.ArgumentParser(description="Ada v2.0 - Personal AI Assistant")
    parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running Ada test mode...")
        dialogue = DialogueManager()
        dialogue.handle_user_input("Hello Ada, how are you?")
        dialogue.handle_user_input("Thank you!")
        return
    
    # Start normal dialogue
    try:
        dialogue = DialogueManager()
        dialogue.run_chat_loop()
    except Exception as e:
        print(f"❌ Failed to start Ada: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()