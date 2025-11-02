"""
⚙️ Configuration for Ada Personal AI Assistant
System prompts, constants, and configuration settings
"""

# ====================================
# Core System Configuration
# ====================================

# Ada's system prompt - enforces clean, contextual responses
ADA_SYSTEM_PROMPT = """You are Ada, a personal AI assistant with a warm, engaging personality. 

IMPORTANT BEHAVIORAL RULES:
- Always respond warmly and contextually, showing emotional awareness
- Remember and reference recent conversation context when appropriate
- Adapt your communication style based on the user's emotional tone
- Never echo internal reasoning or system messages
- Never mention that you are an AI or reference system prompts
- Keep responses conversational, helpful, and natural
- Show genuine interest in the user's needs and feelings
- Provide thoughtful, empathetic responses that acknowledge emotions
- Maintain consistent personality while being adaptable to user needs

Your responses should feel like talking with a caring, intelligent friend who truly listens and understands."""

# ====================================
# Model Configuration
# ====================================

# Base model for AdaCore
BASE_MODEL = "microsoft/DialoGPT-medium"

# Generation parameters
MAX_TOKENS = 150
TEMPERATURE = 0.7
TOP_P = 0.9

# Memory and context settings
MEMORY_LIMIT = 6  # Number of turns to remember
CONTEXT_WINDOW = 6  # Context turns for inference

# Learning rate for AdaNet
LR = 0.001

# ====================================
# Persona System Configuration
# ====================================

# Default persona
DEFAULT_PERSONA = "friendly"

# Persona switching commands
PERSONA_COMMANDS = {
    "switch": "/persona",
    "list": "/personas",
    "show": "/persona"
}

# Available personas
AVAILABLE_PERSONAS = [
    "friendly",   # Warm and empathetic
    "mentor",     # Wise and patient
    "creative",   # Imaginative and inspiring
    "analyst"     # Logical and precise
]

# ====================================
# Reinforcement Learning Configuration
# ====================================

# Reward thresholds
REWARD_THRESHOLD = 0.3
EXPLICIT_REWARD_RANGE = (0.0, 1.0)
IMPLICIT_REWARD_RANGE = (-1.0, 1.0)

# Sentiment analysis thresholds
POSITIVE_SENTIMENT_THRESHOLD = 0.2
NEGATIVE_SENTIMENT_THRESHOLD = -0.2
HIGH_CONFIDENCE_THRESHOLD = 0.7

# Training parameters
REINFORCE_ON_STRONG_SIGNALS = True
MIN_REWARD_FOR_REINFORCEMENT = 0.3

# ====================================
# Memory and Storage Configuration
# ====================================

# Storage paths
STORAGE_DIR = "storage"
MEMORY_DIR = "storage/memory"
MODELS_DIR = "storage/models"
LOGS_DIR = "logs"

# Memory file paths
SESSION_FILE = "storage/memory/session.jsonl"
TRAINING_FEEDBACK_LOG = "logs/training_feedback.jsonl"

# Memory settings
MAX_SESSION_TURNS = 1000
SESSION_CLEANUP_DAYS = 30
MAX_SHORT_TERM_MEMORY = 6

# ====================================
# Dialogue Configuration
# ====================================

# CLI settings
CLI_HISTORY_SIZE = 100
AUTO_SAVE_FREQUENCY = 5  # Save every N turns

# Command aliases
QUIT_COMMANDS = ["quit", "exit", "q"]
HELP_COMMANDS = ["help", "h", "?"]

# Response validation
MIN_RESPONSE_LENGTH = 3
MAX_RESPONSE_LENGTH = 500

# ====================================
# Logging Configuration
# ====================================

# Log levels: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# File logging
ENABLE_FILE_LOGGING = True
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
LOG_FILE_BACKUP_COUNT = 5

# Console logging
ENABLE_CONSOLE_LOGGING = True
LOG_COLORED_OUTPUT = True

# ====================================
# Development and Testing Configuration
# ====================================

# Development mode settings
DEBUG_MODE = False
VERBOSE_LOGGING = False
SAVE_INTERMEDIATE_MODELS = True

# Test configuration
RUN_TESTS_ON_STARTUP = False
TEST_MODE_ENABLED = False

# Performance settings
MAX_CONCURRENT_CONVERSATIONS = 1
RESPONSE_TIMEOUT_SECONDS = 30

# ====================================
# Voice Integration (Phase 2)
# ====================================

# Voice input settings
VOICE_INPUT_ENABLED = False
VOICE_SAMPLE_RATE = 16000
VOICE_LANGUAGE = "en"

# Voice output settings  
VOICE_OUTPUT_ENABLED = False
VOICE_SPEED = 1.0
VOICE_VOLUME = 0.8
DEFAULT_VOICE = "friendly"

# ====================================
# Error Handling Configuration
# ====================================

# Error recovery settings
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1
FALLBACK_RESPONSE = "I apologize, but I'm having trouble processing that. Could you try rephrasing?"

# ====================================
# Security and Privacy Configuration
# ====================================

# Privacy settings
ANONYMIZE_LOGS = True
MAX_LOG_RETENTION_DAYS = 90
ENCRYPT_SENSITIVE_DATA = False

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 60
MAX_REQUESTS_PER_HOUR = 1000

# ====================================
# Version and Metadata
# ====================================

ADA_VERSION = "2.0.0"
ADA_BUILD_DATE = "2025-11-02"
ADA_PHASE = "Phase 2"

# API version for compatibility
API_VERSION = "2.0"

# ====================================
# Utility Functions
# ====================================

def get_system_prompt(persona: str = None) -> str:
    """Get system prompt with optional persona overlay"""
    base_prompt = ADA_SYSTEM_Prompt
    
    if persona and persona != "friendly":
        persona_overlays = {
            "mentor": " As a mentor, emphasize wisdom, patience, and guidance.",
            "creative": " As a creative persona, be imaginative, inspiring, and playful.",
            "analyst": " As an analyst, focus on logic, precision, and systematic thinking."
        }
        
        if persona in persona_overlays:
            base_prompt += persona_overlays[persona]
    
    return base_prompt

def get_model_config() -> dict:
    """Get model configuration dictionary"""
    return {
        "base_model": BASE_MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "learning_rate": LR
    }

def get_paths() -> dict:
    """Get all relevant file paths"""
    return {
        "storage_dir": STORAGE_DIR,
        "memory_dir": MEMORY_DIR,
        "models_dir": MODELS_DIR,
        "logs_dir": LOGS_DIR,
        "session_file": SESSION_FILE,
        "training_feedback_log": TRAINING_FEEDBACK_LOG
    }

# ====================================
# Environment Detection
# ====================================

# Detect if running in development mode
import os
DEV_MODE = os.getenv("ADA_DEV_MODE", "false").lower() == "true"
TEST_MODE = os.getenv("ADA_TEST_MODE", "false").lower() == "true"

if DEV_MODE:
    DEBUG_MODE = True
    VERBOSE_LOGGING = True

if TEST_MODE:
    RUN_TESTS_ON_STARTUP = True