"""
🧠 Neural Module for Ada
Contains Ada's brain components and training logic
"""

from .policy_network import AdaCore, create_ada_core
from .encoder import TextEncoder, simple_text_embedding
from .reward_model import RewardModel, RuleBasedReward
from .trainer import AdaTrainer

__all__ = [
    "AdaCore",
    "create_ada_core", 
    "TextEncoder",
    "simple_text_embedding",
    "RewardModel",
    "RuleBasedReward",
    "AdaTrainer"
]