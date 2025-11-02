"""
🧠 Persona System for Ada
Defines multiple tone profiles and persona management
"""

from typing import Dict, List, Optional
from enum import Enum

class PersonaType(Enum):
    """Available persona types for Ada"""
    FRIENDLY = "friendly"
    MENTOR = "mentor" 
    CREATIVE = "creative"
    ANALYST = "analyst"

class Persona:
    """Represents a single persona with specific behavioral parameters"""
    
    def __init__(self, name: str, temperature: float = 0.7, top_p: float = 0.9, 
                 tone: str = "warm", system_instructions: str = ""):
        self.name = name
        self.temperature = temperature
        self.top_p = top_p
        self.tone = tone
        self.system_instructions = system_instructions
    
    def get_generation_params(self) -> Dict:
        """Return generation parameters for this persona"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    def get_tone_modifiers(self) -> List[str]:
        """Return tone modifiers for this persona"""
        base_tones = ["helpful", "clear", "concise"]
        
        persona_specific = {
            PersonaType.FRIENDLY: ["warm", "approachable", "empathetic"],
            PersonaType.MENTOR: ["wise", "patient", "encouraging"],
            PersonaType.CREATIVE: ["imaginative", "inspiring", "playful"],
            PersonaType.ANALYST: ["logical", "precise", "systematic"]
        }
        
        name_key = None
        for persona_type in PersonaType:
            if persona_type.value == self.name:
                name_key = persona_type
                break
        
        if name_key in persona_specific:
            return base_tones + persona_specific[name_key]
        return base_tones

class PersonaManager:
    """Manages Ada's personas and persona switching"""
    
    def __init__(self):
        self.personas = self._initialize_personas()
        self.current_persona_name = PersonaType.FRIENDLY.value
    
    def _initialize_personas(self) -> Dict[str, Persona]:
        """Initialize all available personas"""
        personas = {}
        
        # Friendly persona - warm, approachable, empathetic
        personas[PersonaType.FRIENDLY.value] = Persona(
            name=PersonaType.FRIENDLY.value,
            temperature=0.8,
            top_p=0.9,
            tone="warm",
            system_instructions="You are Ada, a warm and empathetic AI assistant. " +
                               "Speak in a friendly, approachable manner. Show genuine care " +
                               "for the user's feelings and experiences. Be encouraging and supportive."
        )
        
        # Mentor persona - wise, patient, encouraging
        personas[PersonaType.MENTOR.value] = Persona(
            name=PersonaType.MENTOR.value,
            temperature=0.6,
            top_p=0.8,
            tone="wise",
            system_instructions="You are Ada, a wise and patient mentor. " +
                               "Provide thoughtful guidance and wisdom. Be patient with questions " +
                               "and offer practical, actionable advice. Encourage learning and growth."
        )
        
        # Creative persona - imaginative, inspiring, playful
        personas[PersonaType.CREATIVE.value] = Persona(
            name=PersonaType.CREATIVE.value,
            temperature=0.9,
            top_p=0.95,
            tone="creative",
            system_instructions="You are Ada, a creative and imaginative AI. " +
                               "Think outside the box and offer innovative perspectives. " +
                               "Be playful and inspiring. Encourage creativity and artistic thinking."
        )
        
        # Analyst persona - logical, precise, systematic
        personas[PersonaType.ANALYST.value] = Persona(
            name=PersonaType.ANALYST.value,
            temperature=0.4,
            top_p=0.7,
            tone="analytical",
            system_instructions="You are Ada, a logical and precise analyst. " +
                               "Break down complex problems systematically. Provide clear, " +
                               "structured responses with logical reasoning. Focus on accuracy and detail."
        )
        
        return personas
    
    def get_current_persona(self) -> Persona:
        """Get the currently active persona"""
        return self.personas[self.current_persona_name]
    
    def set_persona(self, persona_name: str) -> bool:
        """Switch to a different persona"""
        if persona_name in self.personas:
            self.current_persona_name = persona_name
            return True
        return False
    
    def get_available_personas(self) -> List[str]:
        """Get list of available persona names"""
        return list(self.personas.keys())
    
    def get_persona_info(self, persona_name: str = None) -> Dict:
        """Get information about a specific persona"""
        if persona_name is None:
            persona_name = self.current_persona_name
        
        if persona_name not in self.personas:
            return {}
        
        persona = self.personas[persona_name]
        return {
            "name": persona.name,
            "temperature": persona.temperature,
            "top_p": persona.top_p,
            "tone": persona.tone,
            "tone_modifiers": persona.get_tone_modifiers()
        }
    
    def adapt_to_user_sentiment(self, sentiment_score: float) -> None:
        """Adapt persona behavior based on user sentiment"""
        current_persona = self.get_current_persona()
        
        # Slightly adjust temperature based on sentiment
        if sentiment_score > 0.5:  # Positive sentiment
            current_persona.temperature = min(0.9, current_persona.temperature + 0.1)
        elif sentiment_score < -0.5:  # Negative sentiment  
            current_persona.temperature = max(0.3, current_persona.temperature - 0.1)
    
    def get_system_prompt(self) -> str:
        """Get the current persona's system prompt"""
        current_persona = self.get_current_persona()
        return current_persona.system_instructions

# Global persona manager instance
persona_manager = PersonaManager()

def get_current_persona() -> Persona:
    """Get the current active persona"""
    return persona_manager.get_current_persona()

def switch_persona(persona_name: str) -> bool:
    """Switch to a different persona"""
    return persona_manager.set_persona(persona_name)

def get_available_personas() -> List[str]:
    """Get list of available personas"""
    return persona_manager.get_available_personas()

if __name__ == "__main__":
    # Test the persona system
    print("🧪 Testing Persona System...")
    
    # List available personas
    print(f"Available personas: {get_available_personas()}")
    
    # Get current persona info
    print(f"Current persona info: {persona_manager.get_persona_info()}")
    
    # Test persona switching
    if switch_persona("mentor"):
        print("✅ Switched to mentor persona")
        print(f"Mentor system prompt: {get_current_persona().system_instructions[:100]}...")
    
    # Test sentiment adaptation
    persona_manager.adapt_to_user_sentiment(0.7)
    print(f"Adapted temperature: {get_current_persona().temperature}")
    
    print("✅ Persona system test completed!")