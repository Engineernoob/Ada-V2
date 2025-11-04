"""
🔮 Reflection System for Ada v3.0
Session summarization and self-review capabilities
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SessionTurn:
    """Represents a single turn in a conversation"""
    user_input: str
    ada_response: str
    timestamp: str
    reward: float = 0.0
    sentiment_score: float = 0.0
    topics: List[str] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []

@dataclass
class SessionReflection:
    """Complete session reflection and analysis"""
    session_id: str
    session_start: str
    session_end: str
    total_turns: int
    average_reward: float
    sentiment_trend: str
    topics_discussed: List[str]
    key_insights: List[str]
    personality_observations: List[str]
    improvement_suggestions: List[str]
    reflection_text: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "session_end": self.session_end,
            "total_turns": self.total_turns,
            "average_reward": self.average_reward,
            "sentiment_trend": self.sentiment_trend,
            "topics_discussed": self.topics_discussed,
            "key_insights": self.key_insights,
            "personality_observations": self.personality_observations,
            "improvement_suggestions": self.improvement_suggestions,
            "reflection_text": self.reflection_text
        }

class TopicExtractor:
    """Extracts topics and themes from conversation text"""
    
    def __init__(self):
        self.topic_keywords = {
            "technology": ["programming", "coding", "software", "ai", "machine learning", "neural", "python", "computer"],
            "personal": ["feel", "feeling", "mood", "happy", "sad", "excited", "tired", "work", "family", "life"],
            "learning": ["learn", "study", "understand", "explain", "teach", "help", "question", "curious"],
            "creative": ["create", "write", "art", "music", "design", "imagine", "story", "idea", "innovative"],
            "analysis": ["analyze", "data", "statistics", "pattern", "trend", "compare", "research", "study"],
            "planning": ["plan", "future", "goal", "schedule", "organize", "prepare", "strategy", "decide"],
            "memory": ["remember", "recall", "forget", "past", "experience", "history", "earlier"],
            "emotion": ["love", "hate", "happy", "angry", "frustrated", "excited", "worried", "calm"]
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities and important concepts"""
        # Simple keyword-based entity extraction
        entities = []
        
        # Look for capitalized words (potential proper nouns)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        return entities

class SentimentAnalyzer:
    """Analyzes sentiment trends across sessions"""
    
    def __init__(self):
        self.positive_words = {
            "love", "great", "wonderful", "amazing", "fantastic", "excellent", "happy", "joy", 
            "pleased", "satisfied", "excited", "proud", "confident", "optimistic", "grateful"
        }
        
        self.negative_words = {
            "hate", "terrible", "awful", "bad", "sad", "upset", "frustrated", "angry", 
            "disappointed", "worried", "stressed", "tired", "confused", "difficult"
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (-1 to 1 scale)"""
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def trend_analysis(self, sentiments: List[float]) -> str:
        """Analyze sentiment trend across a session"""
        if len(sentiments) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        early_avg = sum(sentiments[:len(sentiments)//2]) / (len(sentiments)//2)
        late_avg = sum(sentiments[len(sentiments)//2:]) / (len(sentiments) - len(sentiments)//2)
        
        if late_avg > early_avg + 0.1:
            return "improving"
        elif late_avg < early_avg - 0.1:
            return "declining"
        else:
            return "stable"

class ReflectionGenerator:
    """Generates natural language reflections and insights"""
    
    def __init__(self):
        self.topic_extractor = TopicExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def generate_reflection(self, session_turns: List[SessionTurn], session_id: str) -> SessionReflection:
        """Generate complete session reflection"""
        
        if not session_turns:
            raise ValueError("Cannot generate reflection for empty session")
        
        # Extract session metadata
        session_start = session_turns[0].timestamp
        session_end = session_turns[-1].timestamp
        total_turns = len(session_turns)
        
        # Calculate average reward
        rewards = [turn.reward for turn in session_turns if turn.reward != 0]
        average_reward = sum(rewards) / max(len(rewards), 1)
        
        # Analyze sentiment trends
        sentiments = [self.sentiment_analyzer.analyze_sentiment(f"{turn.user_input} {turn.ada_response}") 
                     for turn in session_turns]
        sentiment_trend = self.sentiment_analyzer.trend_analysis(sentiments)
        
        # Extract topics
        all_text = " ".join([f"{turn.user_input} {turn.ada_response}" for turn in session_turns])
        topics_discussed = list(set(self.topic_extractor.extract_topics(all_text)))
        
        # Generate insights
        key_insights = self._generate_key_insights(session_turns, topics_discussed, average_reward)
        personality_observations = self._generate_personality_observations(session_turns, average_reward)
        improvement_suggestions = self._generate_improvement_suggestions(session_turns, sentiment_trend)
        
        # Generate main reflection text
        reflection_text = self._generate_reflection_text(
            session_id, total_turns, average_reward, sentiment_trend, 
            topics_discussed, key_insights
        )
        
        return SessionReflection(
            session_id=session_id,
            session_start=session_start,
            session_end=session_end,
            total_turns=total_turns,
            average_reward=average_reward,
            sentiment_trend=sentiment_trend,
            topics_discussed=topics_discussed,
            key_insights=key_insights,
            personality_observations=personality_observations,
            improvement_suggestions=improvement_suggestions,
            reflection_text=reflection_text
        )
    
    def _generate_key_insights(self, turns: List[SessionTurn], topics: List[str], avg_reward: float) -> List[str]:
        """Generate key insights from the session"""
        insights = []
        
        # Topic-based insights
        if topics:
            main_topics = topics[:3]  # Top 3 topics
            insights.append(f"Primary topics discussed: {', '.join(main_topics)}")
        
        # Reward-based insights
        if avg_reward > 0.7:
            insights.append("Session showed highly positive engagement and satisfaction")
        elif avg_reward > 0.4:
            insights.append("Generally positive interaction with good engagement")
        elif avg_reward > 0.1:
            insights.append("Neutral interaction with moderate engagement")
        else:
            insights.append("Session showed challenges in engagement or satisfaction")
        
        # Length-based insights
        if len(turns) > 10:
            insights.append("Extended conversation indicating strong engagement")
        elif len(turns) > 5:
            insights.append("Moderate-length conversation with sustained interaction")
        
        # Sentiment progression
        sentiments = [self.sentiment_analyzer.analyze_sentiment(f"{t.user_input} {t.ada_response}") for t in turns]
        if len(sentiments) > 2:
            if sentiments[-1] > sentiments[0] + 0.2:
                insights.append("User mood improved during our conversation")
            elif sentiments[-1] < sentiments[0] - 0.2:
                insights.append("User mood became more challenging during interaction")
        
        return insights
    
    def _generate_personality_observations(self, turns: List[SessionTurn], avg_reward: float) -> List[str]:
        """Generate observations about personality and interaction style"""
        observations = []
        
        # Question patterns
        question_count = sum(1 for turn in turns if "?" in turn.user_input)
        if question_count > len(turns) * 0.3:
            observations.append("User shows curiosity and asks many questions")
        
        # Length patterns
        avg_user_length = sum(len(turn.user_input) for turn in turns) / len(turns)
        if avg_user_length > 50:
            observations.append("User provides detailed, expressive inputs")
        elif avg_user_length < 20:
            observations.append("User prefers concise, direct communication")
        
        # Emotional patterns
        emotional_turns = [t for t in turns if any(word in f"{t.user_input} {t.ada_response}".lower() 
                           for word in ["feel", "emotion", "happy", "sad", "excited"])]
        if len(emotional_turns) > 0:
            observations.append("User shares emotional content and personal feelings")
        
        # Reward patterns
        if avg_reward > 0.6:
            observations.append("User appears satisfied with Ada's responses and personality")
        elif avg_reward < 0.3:
            observations.append("User may need different communication approach")
        
        return observations
    
    def _generate_improvement_suggestions(self, turns: List[SessionTurn], sentiment_trend: str) -> List[str]:
        """Generate suggestions for improvement"""
        suggestions = []
        
        # Trend-based suggestions
        if sentiment_trend == "declining":
            suggestions.append("Consider more empathetic and supportive responses")
            suggestions.append("Ask clarifying questions to better understand user needs")
        
        if sentiment_trend == "improving":
            suggestions.append("Continue current approach - user engagement is growing")
        
        # Length-based suggestions
        if len(turns) < 5:
            suggestions.append("Encourage more detailed conversation to build rapport")
        
        # Topic-based suggestions
        all_text = " ".join([f"{t.user_input} {t.ada_response}" for t in turns])
        if "technology" in self.topic_extractor.extract_topics(all_text):
            suggestions.append("Consider providing more technical details when discussing AI/technology")
        
        # Reward-based suggestions
        rewards = [t.reward for t in turns if t.reward != 0]
        if rewards and sum(rewards) / len(rewards) < 0.4:
            suggestions.append("Focus on being more helpful and responsive to user needs")
        
        return suggestions
    
    def _generate_reflection_text(self, session_id: str, total_turns: int, avg_reward: float, 
                                 sentiment_trend: str, topics: List[str], insights: List[str]) -> str:
        """Generate the main reflection text"""
        
        # Date formatting
        session_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        reflection = f"""
**Session Reflection - {session_id}**
Date: {session_date}

**Session Overview**
This session contained {total_turns} conversation turns, focusing on {', '.join(topics[:3]) if topics else 'general discussion'}. 

**Engagement Metrics**
- Average reward score: {avg_reward:.3f}
- Sentiment trend: {sentiment_trend}
- Total interaction time: {(datetime.now() - datetime.fromisoformat(session_date)).total_seconds()/60:.1f} minutes

**Key Insights**
"""
        
        for insight in insights:
            reflection += f"- {insight}\n"
        
        reflection += f"""
**Overall Assessment**
{"This session demonstrated strong user engagement and positive interaction." if avg_reward > 0.6 
 else "This session showed moderate engagement with room for improvement." if avg_reward > 0.3
 else "This session presented challenges that need addressing in future interactions."}

**Learning Points**
- {f"User responds well to the current approach" if avg_reward > 0.6 
  else f"User may prefer more tailored communication style" if avg_reward < 0.3
  else "Current approach is functioning adequately"}

**Next Steps**
- {"Continue building on successful interaction patterns" if avg_reward > 0.6 
  else "Focus on understanding user preferences and adjusting communication style" if avg_reward < 0.3
  else "Monitor and refine interaction approach"}
- {"Explore topics mentioned in more depth" if topics else "Identify more engaging discussion topics"}
- {"Maintain current level of engagement" if sentiment_trend == "stable" 
  else "Build on improving sentiment" if sentiment_trend == "improving"
  else "Address declining sentiment patterns"}

---
*Generated by Ada v3.0 Reflection System*
        """
        
        return reflection.strip()

class ReflectionManager:
    """Manages session reflections and persistent storage"""
    
    def __init__(self, summary_dir: str = "storage/memory/summaries"):
        self.summary_dir = Path(summary_dir)
        self.reflection_generator = ReflectionGenerator()
        
        # Ensure directory exists
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔮 Reflection system initialized")
        print(f"📂 Summary directory: {self.summary_dir}")
    
    def reflect(self, session_history: List[Dict], avg_reward: float = 0.0, session_id: str = None) -> str:
        """
        Main reflection function as specified in Phase-3 requirements
        
        Args:
            session_history: List of conversation turns with 'user_input', 'ada_response', 'timestamp', 'reward'
            avg_reward: Average reward score for the session
            session_id: Session identifier
        
        Returns:
            Natural language reflection text
        """
        if not session_history:
            return "No session data available for reflection."
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert to SessionTurn objects
        session_turns = []
        for turn_data in session_history:
            turn = SessionTurn(
                user_input=turn_data.get('user_input', ''),
                ada_response=turn_data.get('ada_response', ''),
                timestamp=turn_data.get('timestamp', datetime.now().isoformat()),
                reward=turn_data.get('reward', avg_reward),
                sentiment_score=0.0  # Will be calculated
            )
            session_turns.append(turn)
        
        # Generate reflection
        reflection = self.reflection_generator.generate_reflection(session_turns, session_id)
        
        # Save reflection to file
        self._save_reflection(reflection)
        
        return reflection.reflection_text
    
    def _save_reflection(self, reflection: SessionReflection):
        """Save reflection to file"""
        try:
            # Create date-based filename
            date_str = datetime.now().strftime('%Y-%m-%d')
            filename = f"reflection_{reflection.session_id}_{date_str}.json"
            filepath = self.summary_dir / filename
            
            # Save as JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(reflection.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Also save as text file for easy reading
            text_filename = f"reflection_{reflection.session_id}_{date_str}.txt"
            text_filepath = self.summary_dir / text_filename
            
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(reflection.reflection_text)
            
            print(f"📝 Reflection saved: {filepath.name}")
            
        except Exception as e:
            print(f"⚠️ Error saving reflection: {e}")
    
    def get_recent_reflections(self, count: int = 5) -> List[Dict]:
        """Get recent reflection summaries"""
        try:
            reflection_files = list(self.summary_dir.glob("reflection_*.json"))
            reflection_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            recent_reflections = []
            for file_path in reflection_files[:count]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        recent_reflections.append(data)
                except Exception as e:
                    print(f"⚠️ Error reading reflection {file_path}: {e}")
            
            return recent_reflections
            
        except Exception as e:
            print(f"⚠️ Error loading reflections: {e}")
            return []

# Global reflection manager instance
reflection_manager = None

def initialize_reflection_system(summary_dir: str = "storage/memory/summaries") -> ReflectionManager:
    """Initialize the global reflection system"""
    global reflection_manager
    reflection_manager = ReflectionManager(summary_dir)
    return reflection_manager

def reflect(session_history: List[Dict], avg_reward: float = 0.0, session_id: str = None) -> str:
    """Generate session reflection (Phase-3 specification)"""
    if reflection_manager:
        return reflection_manager.reflect(session_history, avg_reward, session_id)
    return "Reflection system not initialized"

if __name__ == "__main__":
    # Test the reflection system
    print("🧪 Testing Reflection System...")
    
    # Initialize
    reflection_mgr = ReflectionManager("test_summaries")
    
    # Create test session data
    test_session = [
        {
            "user_input": "Hello Ada, how are you today?",
            "ada_response": "Hello! I'm doing wonderfully, thank you for asking. How are you?",
            "timestamp": "2025-11-02T20:30:00",
            "reward": 0.8
        },
        {
            "user_input": "I'm excited about the new AI features you're getting!",
            "ada_response": "That's fantastic! I'm really excited about the Phase 3 enhancements too. What aspect interests you most?",
            "timestamp": "2025-11-02T20:31:00",
            "reward": 0.9
        },
        {
            "user_input": "The long-term memory system sounds amazing",
            "ada_response": "Yes! Being able to remember our conversations across sessions is such an exciting development. It will make our interactions much more meaningful.",
            "timestamp": "2025-11-02T20:32:00",
            "reward": 0.7
        }
    ]
    
    # Generate reflection
    reflection_text = reflect(test_session, avg_reward=0.8, session_id="test_session_001")
    
    print("\n📝 Generated Reflection:")
    print("=" * 50)
    print(reflection_text)
    print("=" * 50)
    
    # Test recent reflections
    recent = reflection_mgr.get_recent_reflections(5)
    print(f"\n📊 Recent reflections: {len(recent)}")
    
    print("\n✅ Reflection system test completed!")