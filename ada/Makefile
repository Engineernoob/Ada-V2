# Ada Personal AI Assistant - Makefile
# Phase 1: Local Development Setup

.PHONY: help setup run train test clean install lint check system-info

# Default target
help:
	@echo "🤖 Ada Personal AI Assistant - Makefile"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup     - Install Python dependencies and setup environment"
	@echo "  make run       - Start Ada CLI interface"  
	@echo "  make train     - Run neural network training test"
	@echo "  make test      - Run comprehensive system tests"
	@echo "  make check     - Check system requirements and dependencies"
	@echo "  make clean     - Clean temporary files and caches"
	@echo "  make system-info - Show system information and capabilities"
	@echo ""
	@echo "Phase 2 (Future):"
	@echo "  make install-whisper - Install Whisper.cpp for speech input"
	@echo "  make install-piper   - Install Piper TTS for voice output"

# Setup Python environment and dependencies
setup:
	@echo "🚀 Setting up Ada development environment..."
	@echo ""
	
	# Check Python version
	@python3 --version || (echo "❌ Python 3 not found. Please install Python 3.11+" && exit 1)
	@echo "✅ Python found"
	
	# Create virtual environment if it doesn't exist
	@if [ ! -d ".venv" ]; then \
		echo "📦 Creating Python virtual environment..."; \
		python3 -m venv .venv; \
	else \
		echo "✅ Virtual environment exists"; \
	fi
	
	# Activate virtual environment and upgrade pip
	@echo "⬆️  Upgrading pip..."
	@. .venv/bin/activate && pip install --upgrade pip
	
	# Install dependencies
	@echo "📥 Installing Python dependencies..."
	@. .venv/bin/activate && pip install -r requirements.txt
	
	@echo ""
	@echo "✅ Setup completed! Activate environment with: source .venv/bin/activate"

# Run Ada CLI interface
run:
	@echo "🤖 Starting Ada CLI interface..."
	@echo ""
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && python -m ada.interfaces.cli

# Run neural network training test
train:
	@echo "🧠 Running Ada neural network training test..."
	@echo ""
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && python -m ada.neural.trainer

# Test MPS (Apple Silicon) GPU support
test-mps:
	@echo "🍎 Testing Apple Silicon (MPS) GPU support..."
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	@. .venv/bin/activate && python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Run comprehensive system tests
test: test-mps
	@echo "🧪 Running comprehensive Ada system tests..."
	@echo ""
	
	@if [ ! -d ".venv" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi
	
	@echo "Testing neural components..."
	@. .venv/bin/activate && python -m ada.neural.policy_network
	@echo ""
	@echo "Testing database..."
	@. .venv/bin/activate && python -m ada.storage.conversation_db
	@echo ""
	@echo "Testing CLI..."
	@. .venv/bin/activate && python -m ada.interfaces.cli --test
	@echo ""
	@echo "Testing voice placeholders..."
	@. .venv/bin/activate && python -m ada.interfaces.speech_input
	@. .venv/bin/activate && python -m ada.interfaces.voice_output
	@echo ""
	@echo "✅ All tests completed!"

# Check system requirements and dependencies
check:
	@echo "🔍 Checking Ada system requirements..."
	@echo ""
	@echo "Python version:"
	@python3 --version
	@echo ""
	@echo "Available disk space:"
	@df -h . | tail -1
	@echo ""
	@echo "Virtual environment status:"
	@if [ -d ".venv" ]; then \
		echo "✅ Virtual environment exists"; \
	else \
		echo "❌ Virtual environment not found"; \
	fi
	@echo ""
	@echo "Dependencies status:"
	@if [ -f "requirements.txt" ]; then \
		echo "✅ requirements.txt found"; \
	else \
		echo "❌ requirements.txt missing"; \
	fi
	@echo ""
	@echo "Directory structure:"
	@if [ -d "ada" ]; then \
		echo "✅ ada/ directory exists"; \
	else \
		echo "❌ ada/ directory missing"; \
	fi

# Show system information and capabilities
system-info:
	@echo "💻 Ada System Information"
	@echo "========================"
	@echo ""
	@echo "Operating System: $$(uname -s)"
	@echo "Architecture: $$(uname -m)"
	@echo "Python version:"
	@python3 --version
	@echo ""
	@if [ ! -d ".venv" ]; then \
		echo "⚠️  Virtual environment not set up"; \
	else \
		echo "Virtual environment: ✅ Active"; \
		@echo "Python path: $$(which python3)"; \
	fi
	@echo ""
	@echo "PyTorch and ML capabilities:"
	@if [ ! -d ".venv" ]; then \
		echo "⚠️  Install dependencies first with 'make setup'"; \
	else \
		@. .venv/bin/activate && python -c "import torch; print('PyTorch version:', torch.__version__); print('MPS available:', torch.backends.mps.is_available()); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "⚠️  PyTorch not installed"; \
	fi

# Clean temporary files and caches
clean:
	@echo "🧹 Cleaning Ada temporary files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "✅ Cleanup completed"

# Phase 2 dependencies (future)
install-whisper:
	@echo "🎤 Installing Whisper.cpp (Phase 2)..."
	@echo "This will be implemented in Phase 2 for speech recognition"
	@echo "Planned: git clone https://github.com/ggerganov/whisper.cpp.git"

install-piper:
	@echo "🔊 Installing Piper TTS (Phase 2)..."
	@echo "This will be implemented in Phase 2 for text-to-speech"
	@echo "Planned: pip install piper-tts"

# Development commands
install: setup

# Quick development cycle
dev: clean setup test
	@echo ""
	@echo "🚀 Development environment ready! Run 'make run' to start Ada"