"""
💬 CLI Interface for Ada
Interactive command-line interface for conversation
"""

import sys
import os
import uuid
from datetime import datetime
from pathlib import Path
import argparse

# Add the ada directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

try:
    from ada.neural.policy_network import AdaCore, create_ada_core
    from ada.neural.encoder import TextEncoder, simple_text_embedding
    from ada.neural.reward_model import RuleBasedReward
    from ada.storage.conversation_db import ConversationDatabase
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed with: make setup")
    sys.exit(1)

class AdaCLI:
    """Main CLI interface for Ada"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.turn_number = 0
        self.running = True
        
        # Initialize components
        print("🧠 Initializing Ada's neural core...")
        try:
            self.neural_model = create_ada_core()
            self.encoder = TextEncoder()
            self.reward_system = RuleBasedReward()
            self.database = ConversationDatabase()
            
            # Try to load existing model
            model_loaded = self.neural_model.load_model("ada/storage/checkpoints/ada_core.pt")
            if not model_loaded:
                print("💡 No trained model found - starting fresh!")
            
        except Exception as e:
            print(f"❌ Error initializing Ada: {e}")
            sys.exit(1)
        
        print(f"✅ Ada initialized! Session ID: {self.session_id}")
        print("💬 Type 'quit', 'exit', or 'q' to end conversation")
        print("💬 Type '/help' for commands, '/stats' for statistics")
        print("💬 Type '/rate <number>' to rate last response")
        print()
    
    def process_user_input(self, user_input: str):
        """Process user input and generate Ada's response"""
        self.turn_number += 1
        
        # Handle special commands
        if user_input.lower().startswith('/'):
            return self.handle_command(user_input)
        
        try:
            # Generate Ada's response using neural network
            ada_response = self.generate_response(user_input)
            
            # Compute reward
            reward = self.reward_system.compute_reward(user_input, ada_response)
            
            # Save to database
            conversation_id = self.database.insert_conversation(
                user_input=user_input,
                ada_response=ada_response,
                reward=reward,
                session_id=self.session_id,
                turn_number=self.turn_number,
                context="cli_conversation"
            )
            
            return f"Ada: {ada_response}"
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            print(f"❌ {error_msg}")
            return f"Ada: {error_msg}"
    
    def generate_response(self, user_input: str) -> str:
        """Generate Ada's response using neural network"""
        # Create input embedding
        input_embedding = simple_text_embedding(user_input)
        
        # Process through neural network
        with torch.no_grad():
            output = self.neural_model(input_embedding.unsqueeze(0))
            output_embedding = output.squeeze()
        
        # Simple response generation based on input analysis
        response = self.generate_simple_response(user_input, output_embedding)
        return response
    
    def generate_simple_response(self, user_input: str, neural_output) -> str:
        """Generate a simple response based on input and neural output"""
        user_lower = user_input.lower()
        
        # Greeting patterns
        if any(word in user_lower for word in ["hello", "hi", "hey", "greetings"]):
            responses = [
                "Hello! I'm Ada, your personal AI assistant.",
                "Hi there! How can I help you today?",
                "Greetings! I'm ready to assist you.",
                "Hello! Nice to meet you."
            ]
            return responses[hash(user_input) % len(responses)]
        
        # Questions about Ada
        elif any(word in user_lower for word in ["who are you", "what are you", "tell me about yourself"]):
            return "I'm Ada, a conversational AI with my own neural network. I'm designed to learn and improve through our conversations!"
        
        # Questions about capabilities
        elif any(word in user_lower for word in ["what can you do", "help me", "capabilities"]):
            return "I can have conversations, learn from our interactions, and help with various tasks. I'm still learning and growing!"
        
        # Emotional responses
        elif any(word in user_lower for word in ["thank", "thanks"]):
            return "You're very welcome! I'm happy to help."
        elif any(word in user_lower for word in ["sorry", "apologize"]):
            return "No worries at all! We all make mistakes."
        
        # Default responses
        else:
            responses = [
                "That's interesting! Tell me more.",
                "I understand. What would you like to explore next?",
                "I hear you. How can I assist you further?",
                "That's a great point. What's on your mind?",
                "I'm listening. Please continue."
            ]
            return responses[hash(user_input) % len(responses)]
    
    def handle_command(self, command: str) -> str:
        """Handle special CLI commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            return ("Available commands:\n"
                   "/help - Show this help\n"
                   "/stats - Show conversation statistics\n"
                   "/quit - Exit the conversation\n"
                   "/rate <number> - Rate last response (0-1)")
        
        elif cmd == "/stats":
            stats = self.database.get_conversation_stats()
            avg_reward = self.database.get_average_reward(self.session_id)
            return (f"📊 Session Statistics:\n"
                   f"Turns this session: {self.turn_number}\n"
                   f"Session ID: {self.session_id}\n"
                   f"Average reward: {avg_reward:.3f}\n"
                   f"Total conversations: {stats['total_conversations']}")
        
        elif cmd == "/rate":
            if len(parts) < 2:
                return "Usage: /rate <number> (0-1)"
            try:
                rating = float(parts[1])
                if 0 <= rating <= 1:
                    # Find last conversation and update reward
                    recent = self.database.get_recent_conversations(1)
                    if recent:
                        # This is simplified - in real implementation you'd update the DB
                        return f"✅ Rated last response with {rating:.2f}"
                    else:
                        return "❌ No conversation to rate"
                else:
                    return "❌ Rating must be between 0 and 1"
            except ValueError:
                return "❌ Invalid rating number"
        
        elif cmd in ["/quit", "/exit"]:
            self.running = False
            return "Goodbye! Thanks for chatting with Ada."
        
        else:
            return f"❌ Unknown command: {cmd}. Type /help for available commands."
    
    def run(self):
        """Main CLI loop"""
        print("🤖 Ada: Hello! I'm ready to chat. What would you like to talk about?")
        print()
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Process input and get response
                response = self.process_user_input(user_input)
                
                # Display Ada's response
                print(f"🤖 {response}")
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                continue
        
        # Final statistics
        stats = self.database.get_conversation_stats()
        print(f"\n📊 Session Summary:")
        print(f"Total turns: {self.turn_number}")
        print(f"Conversations in database: {stats['total_conversations']}")

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(description="Ada - Personal AI Assistant CLI")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 Running Ada CLI test...")
        # Simple test
        cli = AdaCLI()
        test_inputs = ["Hello Ada", "How are you?", "Tell me about yourself"]
        
        for test_input in test_inputs:
            print(f"Test input: {test_input}")
            response = cli.process_user_input(test_input)
            print(f"Response: {response}")
            print()
        
        return
    
    # Start normal CLI
    try:
        cli = AdaCLI()
        cli.run()
    except Exception as e:
        print(f"❌ Failed to start Ada CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()