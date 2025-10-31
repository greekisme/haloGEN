# Mira AI Emotion and Reflection Module
# Adds emotional context inference, adaptive tone, and self-reflection.
# Works alongside the Reasoner, Memory, and Logic systems.

import re
import time
import statistics
from memory import memory, update_memory

class EmotionReflector:
    def __init__(self):
        self.emotion_history = []
        self.last_reflection_time = 0
        self.reflection_interval = 60 * 5  # reflect every 5 minutes

    def infer_emotion(self, text: str) -> str:
        """Infer emotional tone from text using simple linguistic cues with regex word boundaries."""
        text = text.lower()

        emotion_keywords = {
            "happy": ["yay", "great", "awesome", "amazing", "good", "nice", "cool", "fun", "excited", "excitement", "wonderful", "fantastic", "brilliant", "excellent"],
            "sad": ["sad", "unhappy", "depressed", "lonely", "upset", "cry", "down", "blue", "gloomy", "melancholy", "disappointed"],
            "angry": ["angry", "mad", "furious", "hate", "annoyed", "rage", "irritated", "frustrated", "livid"],
            "tired": ["tired", "sleepy", "exhausted", "drained", "fatigued", "worn out", "beat", "wiped out"],
            "stressed": ["stressed", "pressure", "worried", "anxious", "nervous", "overwhelmed", "tense", "panicked"],
            "motivated": ["motivated", "productive", "inspired", "focused", "determined", "driven", "ambitious", "energized"]
        }

        for emotion, keywords in emotion_keywords.items():
            for word in keywords:
                # Use regex word boundaries for more precise matching
                if re.search(r'\b' + re.escape(word) + r'\b', text):
                    self.emotion_history.append(emotion)
                    return emotion

        self.emotion_history.append("neutral")
        return "neutral"

    def adjust_tone(self, ai_response: str, emotion: str) -> str:
        """Adjust response tone based on inferred emotion."""
        tone_templates = {
            "sad": "Hey, I get that things can feel off sometimes. ",
            "angry": "Take a deep breath. It's okay to feel upset. ",
            "tired": "You sound a bit drained — maybe take a short break. ",
            "stressed": "Remember, it's okay to pause and breathe. ",
            "motivated": "Love that energy! Keep pushing forward. ",
            "happy": "That's awesome to hear! ",
            "neutral": ""
        }

        prefix = tone_templates.get(emotion, "")
        return prefix + ai_response

    def reflect(self) -> str:
        """Periodically reflect on emotional trends and store insights in memory."""
        now = time.time()
        if now - self.last_reflection_time < self.reflection_interval:
            return ""

        self.last_reflection_time = now
        if not self.emotion_history:
            return "No emotions recorded yet."

        try:
            dominant = statistics.mode(self.emotion_history)
        except statistics.StatisticsError:
            # Handle tie cases by using the most recent emotion
            dominant = self.emotion_history[-1] if self.emotion_history else "neutral"
        
        reflection = f"Lately, I've noticed you've been mostly feeling {dominant}. I'll try to adapt my tone accordingly."
        
        # Store reflection in memory using the memory system
        memory.set("emotional_reflection", reflection, "knowledge")
        
        # Also store the dominant emotion for pattern tracking
        memory.set("dominant_emotion", dominant, "emotions")
        
        return reflection

    def process(self, user_input: str, ai_response: str) -> str:
        """Complete emotion processing pipeline: infer, adjust, reflect."""
        # Infer emotion from user input
        emotion = self.infer_emotion(user_input)
        
        # Adjust AI response tone
        adjusted_response = self.adjust_tone(ai_response, emotion)
        
        # Reflect periodically
        reflection = self.reflect()
        if reflection:
            print(f"[EmotionReflector] Reflection: {reflection}")
        
        return adjusted_response

if __name__ == "__main__":
    print("=== Mira AI Emotion & Reflection Self-Test ===")
    er = EmotionReflector()

    test_inputs = [
        "I'm so tired after studying all day.",
        "I feel great about today's progress!",
        "I'm a bit stressed about my exams.",
        "Ugh I hate when I forget my notes."
    ]

    for text in test_inputs:
        emotion = er.infer_emotion(text)
        adjusted = er.adjust_tone("Let's keep going.", emotion)
        print(f"Input: {text}")
        print(f"→ Inferred Emotion: {emotion}")
        print(f"→ Adjusted Response: {adjusted}")
        print()

    reflection = er.reflect()
    print("Reflection:", reflection)
    print("\n=== Self-Test Complete ===")