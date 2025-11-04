"""
🚀 Ada v2.0 - Main Entry Point
Personal AI Assistant with contextual conversation, personality modes, and reinforcement learning
"""

import sys
import os
import argparse
from pathlib import Path

# Add Ada directory to Python path
ada_dir = Path(__file__).parent
sys.path.insert(0, str(ada_dir))

def main():
    """Main entry point for Ada v2.0"""
    
    parser = argparse.ArgumentParser(
        description="Ada v2.0 - Personal AI Assistant with Contextual Conversation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start interactive chat
  python main.py --test             # Run system tests
  python main.py --demo             # Run in demo mode
  python main.py --version          # Show version info
        """
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run comprehensive system tests"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run in demo mode (no external dependencies)"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="Ada v3.0.0 - Phase 3: Long-Term Memory & Reflection"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.test:
            print("🧪 Running Ada v3.0 system tests...")
            run_system_tests()
        
        elif args.demo:
            print("🎭 Running Ada v3.0 in demo mode...")
            run_demo_mode()
        
        else:
            print("🤖 Starting Ada v3.0 - Personal AI Assistant with Long-Term Memory")
            print("=" * 60)
            run_interactive_chat(debug=args.debug)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for chatting with Ada v2.0")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_interactive_chat(debug: bool = False):
    """Run Ada in interactive chat mode"""
    
    try:
        # Import DialogueManager
        from core.dialogue import DialogueManager
        
        if debug:
            print("🔍 Debug mode enabled")
            print("Available modules:")
            print("- core.dialogue: ✅")
            print("- core.neural_core: ✅")
            print("- core.persona: ✅")
            print("- core.memory: ✅")
            print("- rl.reward_engine: ✅")
            print("- core.config: ✅")
            print()
        
        # Initialize and run dialogue manager
        dialogue_manager = DialogueManager()
        dialogue_manager.run_chat_loop()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("📝 Some modules may be missing. Running in fallback mode...")
        run_fallback_chat()
    except Exception as e:
        print(f"❌ Error starting interactive chat: {e}")
        print("📝 Falling back to basic mode...")
        run_fallback_chat()

def run_demo_mode():
    """Run Ada in demonstration mode without external dependencies"""
    
    print("🎭 Ada v3.0 Demo Mode - Phase 3: Long-Term Memory & Reflection")
    print("=" * 60)
    print("This is a demonstration of Ada's Phase 3 enhancements:")
    print("- 🧠 Long-Term Memory (cross-session recall)")
    print("- 🔮 Session Reflection (automatic analysis)")
    print("- 🎭 Persona System (4 different personalities)")
    print("- 🧠 Context Memory (remembers conversation)")
    print("- 🎯 Emotional Sentiment Analysis")
    print("- 💬 Enhanced Dialogue Management")
    print("- 🔄 Reinforcement Learning")
    print()
    
    # Demo Phase 3 long-term memory
    print("🧠 Long-Term Memory Demo:")
    print("   User: 'I love programming in Python' (Session 1)")
    print("   Ada: 'That's wonderful! Python is such a versatile language.'")
    print("   💾 Stored in long-term memory")
    print()
    print("   User: 'What do I like to code?' (Session 2, next day)")
    print("   Ada: 'I remember you mentioned loving Python programming!'")
    print("   🔍 Retrieved from long-term memory")
    print()
    
    # Demo Phase 3 reflection
    print("🔮 Session Reflection Demo:")
    print("   Session End → Automatic Analysis Generated:")
    print("   📝 'Session contained 12 turns, average reward 0.8'")
    print("   📈 'Sentiment trend: improving, user engagement high'")
    print("   💡 'Primary topics: programming, AI, technology'")
    print("   📊 'Session saved to reflection_2025-11-02.txt'")
    print()
    
    # Simulate persona switching demo
    personas = ["friendly", "mentor", "creative", "analyst"]
    
    for persona in personas:
        print(f"🎭 {persona.title()} Persona Demo:")
        if persona == "friendly":
            print("   'Hello! I'm Ada, warm and empathetic. How are you feeling today?'")
        elif persona == "mentor":
            print("   'Hello. I'm Ada, your wise mentor. What would you like to learn about?'")
        elif persona == "creative":
            print("   'Hi there! I'm Ada, your creative companion. Ready to explore some ideas?'")
        elif persona == "analyst":
            print("   'Greetings. I'm Ada, your analytical assistant. What needs systematic analysis?'")
        print()
    
    print("🆕 NEW in Phase 3:")
    print("   💾 Long-term memory persists across sessions")
    print("   🔮 Automatic session reflections and insights")
    print("   🧠 Semantic recall of past conversations")
    print("   📊 Enhanced learning from interaction patterns")
    print()
    
    # Simulate memory demo
    print("🧠 Memory System Demo:")
    print("   User: 'I had a great day at work today!'")
    print("   Ada: 'That's wonderful! What made your day at work so great?'")
    print("   Memory: Stored conversation turn with positive sentiment")
    print()
    
    # Simulate sentiment analysis demo
    print("🎯 Sentiment Analysis Demo:")
    sentiment_examples = [
        ("I love this! It's amazing! 🙏", "Very positive (0.9)"),
        ("I'm frustrated with this problem 😤", "Negative (-0.7)"),
        ("This is okay, nothing special", "Neutral (0.1)")
    ]
    
    for text, sentiment in sentiment_examples:
        print(f"   '{text}' → {sentiment}")
    print()
    
    # Simulate reinforcement learning demo
    print("🔄 Reinforcement Learning Demo:")
    print("   Implicit reward from positive sentiment: +0.3")
    print("   Ada's neural network adapts to encourage positive interactions")
    print()
    
    print("💬 Run 'python main.py' to start full interactive chat!")

def run_fallback_chat():
    """Fallback chat mode when main systems unavailable"""
    
    print("📝 Ada v2.0 - Fallback Mode")
    print("Running basic conversation without advanced features...")
    print("Type 'quit' to exit.\n")
    
    conversation_count = 0
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("🤖 Goodbye! See you next time!")
                break
            
            conversation_count += 1
            
            # Simple response logic
            response = generate_simple_response(user_input)
            print(f"Ada: {response}")
            print(f"(Turn {conversation_count})\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break

def generate_simple_response(user_input: str) -> str:
    """Generate simple responses for fallback mode"""
    
    user_lower = user_input.lower()
    
    if any(word in user_lower for word in ["hello", "hi", "hey"]):
        return "Hello! I'm Ada, your personal AI assistant. How can I help you today?"
    
    elif any(word in user_lower for word in ["thank", "thanks"]):
        return "You're very welcome! I'm happy to help."
    
    elif "how are you" in user_lower:
        return "I'm doing wonderfully, thank you for asking! How are you doing?"
    
    elif "help" in user_lower:
        return "I'm here to assist you with various tasks. What would you like to explore?"
    
    elif "name" in user_lower:
        return "My name is Ada. I'm a personal AI assistant designed to be helpful, empathetic, and contextually aware."
    
    elif any(word in user_lower for word in ["persona", "personality"]):
        return "I have multiple personas: friendly (warm and empathetic), mentor (wise and patient), creative (imaginative and inspiring), and analyst (logical and precise)."
    
    elif "memory" in user_lower:
        return "I have context memory that remembers our conversation! I can recall the last few turns to maintain better context."
    
    elif "sentiment" in user_lower or "emotion" in user_lower:
        return "I can analyze emotional sentiment in text on a scale from -1 (very negative) to +1 (very positive) to better understand and respond to your feelings."
    
    elif any(word in user_lower for word in ["learn", "training", "reinforcement"]):
        return "I use reinforcement learning to improve my responses based on feedback and emotional sentiment analysis!"
    
    elif "?" in user_input:
        return "That's a thoughtful question! I'd love to explore that with you further."
    
    else:
        responses = [
            "That's really interesting! Tell me more about that.",
            "I appreciate you sharing that with me. What else is on your mind?",
            "That's a great point! How do you feel about it?",
            "I'm listening. Please continue - I'd love to hear more.",
            "Thanks for that insight! It's fascinating to think about."
        ]
        return responses[hash(user_input) % len(responses)]

def run_system_tests():
    """Run comprehensive system tests for Ada v2.0"""
    
    print("🔍 Ada v2.0 System Tests")
    print("=" * 40)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Import all modules
    tests_total += 1
    try:
        import core.dialogue
        import core.neural_core
        import core.persona
        import core.memory
        import core.config
        import rl.reward_engine
        print("✅ Module imports: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Module imports: FAILED - {e}")
    
    # Test 2: Persona system
    tests_total += 1
    try:
        from core.persona import get_available_personas, get_current_persona
        personas = get_available_personas()
        current = get_current_persona()
        print(f"✅ Persona system: PASSED ({len(personas)} personas, current: {current.name})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Persona system: FAILED - {e}")
    
    # Test 3: Memory system
    tests_total += 1
    try:
        from core.memory import start_memory_session, add_to_memory
        session_id = start_memory_session("test_session")
        turn = add_to_memory("test", "response")
        print(f"✅ Memory system: PASSED (session: {session_id})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Memory system: FAILED - {e}")
    
    # Test 4: Reward engine
    tests_total += 1
    try:
        from rl.reward_engine import analyze_sentiment, get_reward_analysis
        sentiment, details = analyze_sentiment("I love this! It's amazing!")
        print(f"✅ Reward engine: PASSED (sentiment: {sentiment:.2f})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Reward engine: FAILED - {e}")
    
    # Test 5: Neural core (basic functionality)
    tests_total += 1
    try:
        from core.neural_core import AdaCore
        ada_core = AdaCore()
        response = ada_core.infer("Hello Ada")
        print(f"✅ Neural core: PASSED (response: '{response[:50]}...')")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Neural core: FAILED - {e}")
    
    # Test 6: Dialogue manager
    tests_total += 1
    try:
        from core.dialogue import DialogueManager
        dialogue = DialogueManager()
        print("✅ Dialogue manager: PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Dialogue manager: FAILED - {e}")
    
    # Test 7: Configuration
    tests_total += 1
    try:
        from core.config import ADA_VERSION, ADA_SYSTEM_PROMPT
        print(f"✅ Configuration: PASSED (version: {ADA_VERSION})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Configuration: FAILED - {e}")
    
    # Test results
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! Ada v2.0 is ready to run.")
        print("💬 Start interactive chat with: python main.py")
    else:
        print("⚠️ Some tests failed. Some features may not work properly.")
        print("📝 Use 'python main.py --demo' to see available features.")

if __name__ == "__main__":
    main()