"""
💬 Interface Module for Ada
CLI and event handling interfaces
"""

from .cli import AdaCLI
from .event_loop import EventLoop, AdaEventManager, EventType, AdaEvent
from .speech_input import SpeechInput
from .voice_output import VoiceOutput

__all__ = [
    "AdaCLI",
    "EventLoop", 
    "AdaEventManager",
    "EventType",
    "AdaEvent",
    "SpeechInput",
    "VoiceOutput"
]