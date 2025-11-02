# 🤖 Ada - Personal AI Assistant

**Phase 1: Local Development Setup for macOS**

Ada is a conversational AI with her own neural network and reinforcement learning core — not dependent on cloud APIs. This first phase sets up a working local architecture and runnable scaffolding that can train and converse in CLI mode.

## 🎯 Phase 1 Objectives

✅ **Scaffolded directory structure and placeholder files**  
✅ **Functional Python environment that runs cleanly on macOS**  
✅ **Small test neural network (PyTorch MLP) as Ada's "brain seed"**  
✅ **CLI loop for typing to Ada**  
✅ **Lightweight persistence (SQLite) for conversations and state**  
✅ **Makefile and requirements.txt for easy setup**  
✅ **Prepared for voice integration (Whisper.cpp, Piper) - Phase 2**

## 📁 Project Structure

```
ada/
├── neural/                     # 🧠 Neural network components
│   ├── policy_network.py      # AdaCore MLP implementation
│   ├── encoder.py             # Text encoding and embeddings
│   ├── reward_model.py        # RL reward system
│   ├── trainer.py             # Training loops and utilities
│   └── __init__.py
├── rl/                        # 🎯 Reinforcement Learning
│   ├── environment.py         # Conversational environment
│   ├── agent.py              # RL agent implementation
│   ├── memory_buffer.py      # Experience replay buffer
│   └── __init__.py
├── core/                      # 🔧 Core reasoning (Phase 2)
│   ├── reasoning.py          # (Future implementation)
│   ├── context_manager.py    # (Future implementation)
│   ├── memory.py             # (Future implementation)
│   └── persona.yaml          # (Future implementation)
├── interfaces/                # 💬 User interfaces
│   ├── cli.py                # Command-line interface
│   ├── event_loop.py         # Async event management
│   ├── speech_input.py       # (Phase 2 placeholder)
│   ├── voice_output.py       # (Phase 2 placeholder)
│   └── __init__.py
├── storage/                   # 💾 Data persistence
│   ├── conversations.db      # SQLite database
│   ├── checkpoints/          # Model checkpoints
│   ├── embeddings.db         # (Future embeddings)
│   └── conversation_db.py    # Database management
├── config/                    # ⚙️ Configuration
│   ├── settings.yaml         # Main configuration
│   └── .env.example          # Environment variables template
├── Makefile                   # Build and run commands
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- **macOS** (M2/M3 or Intel)
- **Python 3.11+**
- **Command Line Tools**

### Installation Steps

1. **Clone/Setup the project directory:**
   ```bash
   # If starting fresh, ensure you're in the ada project directory
   cd /path/to/ada-project
   ```

2. **Set up the Python environment:**
   ```bash
   make setup
   ```

3. **Run Ada CLI:**
   ```bash
   make run
   ```

4. **Test neural network training:**
   ```bash
   make train
   ```

## 🔧 Makefile Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install Python dependencies and setup environment |
| `make run` | Start Ada CLI interface |
| `make train` | Run neural network training test |
| `make test` | Run comprehensive system tests |
| `make check` | Check system requirements |
| `make clean` | Clean temporary files |
| `make system-info` | Show system information and capabilities |

## 💬 Using Ada CLI

Once Ada is running with `make run`, you can:

- **Type conversations** and Ada will respond
- **Use special commands:**
  - `/help` - Show available commands
  - `/stats` - Show conversation statistics  
  - `/quit` - Exit the conversation
  - `/rate <number>` - Rate last response (0-1)

**Example conversation:**
```
You: Hello Ada
Ada: Hello! I'm Ada, your personal AI assistant. How can I help you today?

You: How are you?
Ada: I'm doing well, thank you for asking! I feel great and ready to chat.

You: Tell me about yourself
Ada: I'm Ada, a conversational AI with my own neural network. I'm designed to learn and improve through our conversations!
```

## 🧠 Neural Network Details

### AdaCore MLP
- **Input dimension:** 512
- **Hidden dimension:** 256  
- **Output dimension:** 512
- **Device support:** Auto-detect (MPS for Apple Silicon, CUDA, or CPU)
- **Model size:** Under 1MB (as specified)

### Training Process
- **Generates dummy data** for testing
- **Saves model weights** to `ada/storage/checkpoints/ada_core.pt`
- **Includes checkpoint system** for resuming training

### macOS Optimizations
- **Apple Silicon (MPS) support** with automatic fallback
- **GPU acceleration** when available
- **Test script included:** `python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"`

## 💾 Database Features

### SQLite Conversation Storage
- **Schema:** id, timestamp, user_input, ada_response, reward, session_id, turn_number, context, metadata
- **Automatic indexing** for performance
- **Statistics tracking** (total conversations, average reward, recent activity)
- **Session management** with unique session IDs
- **Auto-cleanup** of old conversations (configurable)

## 🎯 Reinforcement Learning

### Phase 1 Features
- **Simple Q-learning agent** for basic RL
- **Rule-based reward system** with contextual bonuses
- **Experience replay buffer** for training stability
- **Environment simulation** for training scenarios

### Reward System
- **Base reward:** 0.5
- **Greeting bonuses** for proper social interactions
- **Length optimization** for response quality
- **Context-aware responses** for different conversation types

## 🔊 Voice Integration (Phase 2 Preview)

### Placeholder Classes
- **SpeechInput:** Ready for Whisper.cpp integration
- **VoiceOutput:** Ready for Piper TTS integration
- **Both include:** Language support, voice selection, audio device management

### Phase 2 Plans
- **Speech recognition** using Whisper.cpp (local, offline)
- **Text-to-speech** using Piper (lightweight, high quality)
- **Voice conversation mode** (speech-to-speech)

## ⚙️ Configuration

### YAML Settings (`ada/config/settings.yaml`)
- **Neural network parameters**
- **Training configuration**
- **Database settings**
- **Interface preferences**
- **macOS optimizations**

### Environment Variables (`ada/config/.env.example`)
- **Copy to `.env` for customization**
- **Overrides YAML settings**
- **Development and debug options**

## 🧪 Testing

### Automated Tests
```bash
# Run all tests
make test

# Test specific components
python -m ada.neural.policy_network    # Neural network
python -m ada.storage.conversation_db  # Database
python -m ada.interfaces.cli --test    # CLI interface
```

### Manual Testing
```bash
# Test MPS support
make test-mps

# Check system requirements
make check

# Show system info
make system-info
```

## 📊 Performance

### Current Capabilities
- **Model size:** < 1MB (Phase 1 requirement met)
- **Training time:** Seconds for basic test
- **Response time:** < 100ms for simple responses
- **Memory usage:** Minimal (< 100MB baseline)

### macOS Optimizations
- **Apple Silicon GPU acceleration** (MPS backend)
- **Automatic device detection** (MPS > CUDA > CPU)
- **Path handling** using pathlib for cross-platform compatibility

## 🔧 Development

### Adding New Features
1. **Neural network changes:** Edit files in `ada/neural/`
2. **Interface improvements:** Modify `ada/interfaces/`
3. **Database enhancements:** Update `ada/storage/`
4. **Configuration:** Adjust `ada/config/settings.yaml`

### Testing Your Changes
```bash
# Clean build
make clean

# Setup fresh environment  
make setup

# Run tests
make test

# Start development
make run
```

## 🛠️ Troubleshooting

### Common Issues

**Python not found:**
```bash
# Install Python 3.11+ via Homebrew
brew install python@3.11
```

**MPS not available:**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.backends.mps.is_available())"

# Reinstall PyTorch if needed
pip install torch torchvision torchaudio
```

**Import errors:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
make setup
```

**Database errors:**
```bash
# Check storage directory permissions
ls -la ada/storage/

# Reset database (loses conversation history)
rm ada/storage/conversations.db
```

### System Requirements Check
```bash
make system-info
```

## 📈 Phase 2 Roadmap

### Planned Features
- **Voice integration** (Whisper.cpp + Piper)
- **Advanced neural architectures** 
- **Multi-modal capabilities** (vision, audio)
- **Personality learning system**
- **External model integration**
- **Cloud sync** (optional)

### Phase 2 Dependencies (Future)
```bash
# Speech recognition (Whisper.cpp)
make install-whisper

# Text-to-speech (Piper)
make install-piper
```

## 📝 License

This project is for personal use and development. See individual component licenses for PyTorch, transformers, and other dependencies.

## 🤝 Contributing

This is a personal project for Ada's development. Future contributions welcome as the project evolves through phases.

## 📞 Support

For issues related to:
- **Setup problems:** Check `make check` output
- **Training issues:** Review `ada/storage/checkpoints/`
- **CLI problems:** Use `/help` and `/stats` commands
- **Database issues:** Check SQLite logs and permissions

---

**Ada Phase 1 - Complete!** 🎉

Ready for local conversational AI development on macOS with room for growth into Phase 2 voice integration and beyond.