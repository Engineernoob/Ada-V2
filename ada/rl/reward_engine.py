"""
🎯 Reinforcement Learning Reward Engine for Ada
Analyzes emotional sentiment and generates implicit feedback
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class SentimentAnalyzer:
    """Analyzes emotional sentiment in text on a +1 to -1 scale"""
    
    def __init__(self):
        # Define sentiment word lists
        self.positive_words = {
            # Strong positive emotions
            "love", "adore", "amazing", "wonderful", "fantastic", "excellent", 
            "brilliant", "outstanding", "perfect", "incredible", "awesome",
            
            # Medium positive emotions
            "good", "great", "nice", "happy", "joy", "pleased", "satisfied",
            "excited", "enthusiastic", "optimistic", "confident", "proud",
            
            # Mild positive emotions
            "like", "enjoy", "fine", "okay", "alright", "better", "improved",
            "helpful", "useful", "interesting", "cool", "fun"
        }
        
        self.negative_words = {
            # Strong negative emotions
            "hate", "terrible", "awful", "horrible", "disgusting", "frustrated",
            "angry", "furious", "devastated", "heartbroken", "depressed",
            
            # Medium negative emotions
            "bad", "sad", "upset", "worried", "anxious", "stressed", "concerned",
            "disappointed", "frustrated", "confused", "lost", "lonely",
            
            # Mild negative emotions
            "boring", "tired", "annoyed", "irritated", "meh", "unhappy",
            "difficult", "hard", "problem", "issue", "worry", "concern"
        }
        
        # Emotional indicators (context matters)
        self.positive_indicators = [
            r":\)", r":D", r":P", r"😊", r"😄", r"😃", r"👍", r"❤️", r"💕",
            "thank you", "thanks", "grateful", "appreciate", "well done",
            "good job", "excellent work", "awesome job"
        ]
        
        self.negative_indicators = [
            r":\(", r"D:", r":/", r"😢", r"😭", r"😞", r"😔", r"👎",
            "sorry", "excuse me", "my bad", "apologies", "forgive me"
        ]
        
        # Negation patterns that flip sentiment
        self.negation_patterns = [
            r"\bnot\b", r"\bn't\b", r"\bnever\b", r"\bno\b", 
            r"\bhardly\b", r"\bbarely\b", r"\bwithout\b"
        ]
    
    def analyze_sentiment(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze sentiment of text and return score +1 to -1 with details
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Base score
        positive_score = 0
        negative_score = 0
        
        # Check for sentiment words
        for word in words:
            if word in self.positive_words:
                positive_score += 1
            elif word in self.negative_words:
                negative_score += 1
        
        # Check for emoji/indicator patterns
        for pattern in self.positive_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                positive_score += 0.5
        
        for pattern in self.negative_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                negative_score += 0.5
        
        # Handle negations
        negation_count = 0
        for pattern in self.negation_patterns:
            negation_count += len(re.findall(pattern, text_lower))
        
        # Calculate base sentiment score
        net_sentiment = (positive_score - negative_score) / max(len(words), 1)
        
        # Apply negation effects
        if negation_count > 0 and net_sentiment != 0:
            net_sentiment *= (-1) ** negation_count
        
        # Cap the score to [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, net_sentiment))
        
        # Create analysis details
        details = {
            "positive_words_found": positive_score,
            "negative_words_found": negative_score,
            "negation_count": negation_count,
            "word_count": len(words),
            "net_sentiment_raw": net_sentiment,
            "confidence": self._calculate_confidence(positive_score, negative_score, len(words))
        }
        
        return sentiment_score, details
    
    def _calculate_confidence(self, positive_score: float, negative_score: float, word_count: int) -> float:
        """Calculate confidence in sentiment analysis"""
        total_sentiment_words = positive_score + negative_score
        
        # Higher confidence with more sentiment words relative to total words
        word_ratio = total_sentiment_words / max(word_count, 1)
        
        # Higher confidence with clearer sentiment (less balanced)
        balance_ratio = abs(positive_score - negative_score) / max(total_sentiment_words, 1)
        
        # Combine factors
        confidence = (word_ratio * 0.6) + (balance_ratio * 0.4)
        return min(1.0, confidence)

class RewardEngine:
    """Main reward engine that combines explicit and implicit feedback"""
    
    def __init__(self, log_file: str = "logs/training_feedback.jsonl"):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.log_file = Path(log_file)
        self.explicit_rewards_history = []
        self.implicit_rewards_history = []
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        """Ensure the log directory exists"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, user_input: str, ada_response: str = "") -> Dict:
        """
        Analyze user input for emotional sentiment and generate feedback
        Returns comprehensive analysis including sentiment score and suggestions
        """
        sentiment_score, sentiment_details = self.sentiment_analyzer.analyze_sentiment(user_input)
        
        # Generate implicit reward based on sentiment
        implicit_reward = self._generate_implicit_reward(sentiment_score, sentiment_details)
        
        # Assess Ada's response quality (simplified)
        response_quality = self._assess_response_quality(ada_response)
        
        # Combine explicit and implicit feedback
        combined_analysis = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ada_response": ada_response,
            "sentiment_analysis": {
                "score": sentiment_score,
                "details": sentiment_details,
                "interpretation": self._interpret_sentiment(sentiment_score)
            },
            "implicit_reward": implicit_reward,
            "response_quality": response_quality,
            "suggested_improvements": self._suggest_improvements(sentiment_score, response_quality)
        }
        
        # Log the analysis
        self._log_feedback(combined_analysis)
        
        return combined_analysis
    
    def _generate_implicit_reward(self, sentiment_score: float, sentiment_details: Dict) -> float:
        """
        Generate implicit reward based on sentiment analysis
        +1 for very positive, -1 for very negative, 0 for neutral
        """
        # Base reward on sentiment score
        implicit_reward = sentiment_score
        
        # Adjust based on confidence
        confidence = sentiment_details.get("confidence", 0.5)
        adjusted_reward = implicit_reward * confidence
        
        # Store in history
        self.implicit_rewards_history.append(adjusted_reward)
        
        return max(-1.0, min(1.0, adjusted_reward))
    
    def _assess_response_quality(self, ada_response: str) -> float:
        """Assess Ada's response quality (simplified heuristic)"""
        if not ada_response:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length assessment
        if 10 <= len(ada_response) <= 200:
            score += 0.1
        elif len(ada_response) < 5:
            score -= 0.2
        
        # Greeting detection
        if any(greet in ada_response.lower() for greet in ["hello", "hi", "hey"]):
            score += 0.1
        
        # Question handling
        if ada_response.endswith("?"):
            score += 0.1
        
        # Response appropriateness
        response_lower = ada_response.lower()
        if any(word in response_lower for word in ["help", "assist", "support"]):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _interpret_sentiment(self, score: float) -> str:
        """Convert sentiment score to human-readable interpretation"""
        if score > 0.5:
            return "Very positive"
        elif score > 0.2:
            return "Positive"
        elif score > -0.2:
            return "Neutral"
        elif score > -0.5:
            return "Negative"
        else:
            return "Very negative"
    
    def _suggest_improvements(self, sentiment_score: float, response_quality: float) -> List[str]:
        """Generate suggestions for improvement based on analysis"""
        suggestions = []
        
        if sentiment_score < -0.5:
            suggestions.append("User seems upset - consider more empathetic response")
        elif sentiment_score > 0.5:
            suggestions.append("User is positive - match their enthusiasm")
        
        if response_quality < 0.6:
            suggestions.append("Response could be more helpful or detailed")
        
        if not suggestions:
            suggestions.append("Continue current approach - responses are appropriate")
        
        return suggestions
    
    def log_explicit_reward(self, user_input: str, ada_response: str, reward: float):
        """Log explicit user feedback (e.g., /rate command)"""
        explicit_feedback = {
            "timestamp": datetime.now().isoformat(),
            "type": "explicit",
            "user_input": user_input,
            "ada_response": ada_response,
            "reward_score": reward,
            "feedback_source": "user_rating"
        }
        
        self.explicit_rewards_history.append(reward)
        self._log_raw_feedback(explicit_feedback)
    
    def get_feedback_summary(self, window_size: int = 10) -> Dict:
        """Get summary of recent feedback"""
        recent_implicit = self.implicit_rewards_history[-window_size:]
        recent_explicit = self.explicit_rewards_history[-window_size:]
        
        return {
            "implicit_feedback": {
                "recent_average": sum(recent_implicit) / len(recent_implicit) if recent_implicit else 0.0,
                "recent_count": len(recent_implicit),
                "total_count": len(self.implicit_rewards_history)
            },
            "explicit_feedback": {
                "recent_average": sum(recent_explicit) / len(recent_explicit) if recent_explicit else 0.0,
                "recent_count": len(recent_explicit),
                "total_count": len(self.explicit_rewards_history)
            },
            "sentiment_trend": self._analyze_sentiment_trend()
        }
    
    def _analyze_sentiment_trend(self) -> str:
        """Analyze sentiment trend over recent interactions"""
        if len(self.implicit_rewards_history) < 3:
            return "Insufficient data"
        
        recent = self.implicit_rewards_history[-5:]  # Last 5 interactions
        if len(recent) < 2:
            return "Insufficient data"
        
        # Simple trend analysis
        if recent[-1] > recent[0]:
            return "Improving sentiment"
        elif recent[-1] < recent[0]:
            return "Declining sentiment"
        else:
            return "Stable sentiment"
    
    def _log_feedback(self, analysis: Dict):
        """Log comprehensive feedback analysis"""
        self._log_raw_feedback(analysis)
    
    def _log_raw_feedback(self, feedback: Dict):
        """Log raw feedback to JSONL file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error logging feedback: {e}")

# Global reward engine instance
reward_engine = RewardEngine()

def analyze_sentiment(text: str) -> Tuple[float, Dict]:
    """Analyze sentiment of text"""
    return reward_engine.sentiment_analyzer.analyze_sentiment(text)

def get_reward_analysis(user_input: str, ada_response: str = "") -> Dict:
    """Get comprehensive reward analysis"""
    return reward_engine.analyze(user_input, ada_response)

def log_user_rating(user_input: str, ada_response: str, rating: float):
    """Log explicit user rating"""
    reward_engine.log_explicit_reward(user_input, ada_response, rating)

if __name__ == "__main__":
    # Test the reward engine
    print("🧪 Testing Reward Engine...")
    
    # Test sentiment analysis
    test_inputs = [
        "I love this! It's amazing and wonderful!",
        "I'm really frustrated with this problem",
        "This is okay, nothing special",
        "Thank you so much, you're incredibly helpful!",
        "I hate this, it's terrible and awful"
    ]
    
    for user_input in test_inputs:
        analysis = get_reward_analysis(user_input, "Ada response here")
        print(f"\nInput: {user_input}")
        print(f"Sentiment: {analysis['sentiment_analysis']['score']:.3f} ({analysis['sentiment_analysis']['interpretation']})")
        print(f"Implicit reward: {analysis['implicit_reward']:.3f}")
        print(f"Suggestions: {analysis['suggested_improvements']}")
    
    # Test feedback summary
    summary = reward_engine.get_feedback_summary()
    print(f"\nFeedback summary: {summary}")
    
    print("✅ Reward engine test completed!")