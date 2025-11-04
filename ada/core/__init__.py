"""
🧠 Core Module for Ada v3.0
Contains core AI and memory systems
"""

from .config import *
from .neural_core import AdaCore, AdaNet
from .persona import *
from .memory import *
from .dialogue import *
from .long_memory import *
from .reflection import *

__all__ = [
    # Configuration
    "ADA_SYSTEM_PROMPT",
    "ADA_VERSION", 
    "ADA_PHASE",
    
    # Core Systems
    "AdaCore",
    "AdaNet",
    
    # Memory Systems
    "long_memory",
    "reflection",
    "MemoryManager",
    "SessionMemory",
    "ConversationTurn",
    
    # Persona System
    "get_current_persona",
    "get_available_personas",
    "set_persona",
    "Persona",
    
    # Utility functions
    "get_system_prompt",
    "get_model_config", 
    "get_paths"
]