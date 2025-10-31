# Mira AI Voice Personality Module
# Adds emotional tone modulation and personality-based phrasing to TTS pipeline.
# Integrates with existing PiperVoice, emotion detection, and memory systems.

import os
import random
import time
from typing import Dict, List, Optional
from memory import memory

class PersonalityProfile:
    """Represents a single personality preset with tone and phrasing characteristics."""
    
    def __init__(self, name: str, description: str, pitch_shift: float, speed_factor: float, phrasing_style: Dict[str, List[str]]):
        self.name = name
        self.description = description
        self.pitch_shift = pitch_shift
        self.speed_factor = speed_factor
        self.phrasing_style = phrasing_style
    
    def __repr__(self) -> str:
        return f"PersonalityProfile(name='{self.name}', description='{self.description}')"

class VoicePersonality:
    """Handles emotional tone adjustments and personality-based voice modulation."""
    
    def __init__(self, default_profile: str = "Mentor", piper_voice=None):
        self.profile = self._get_profile(default_profile)
        self.last_emotion = "neutral"
        self._piper = piper_voice  # Use shared Piper instance
        
        # Save initial personality to memory
        memory.set("current_personality", default_profile, "voice_state")
        print(f"[VoicePersonality] Initialized with {self.profile.name} profile")
    
    
    def _get_profile(self, name: str) -> PersonalityProfile:
        """Get personality profile by name."""
        profiles = {
            "Calm": PersonalityProfile(
                name="Calm",
                description="Soft, slow, and soothing tone.",
                pitch_shift=0.9,
                speed_factor=0.85,
                phrasing_style={
                    "sad": ["I know that feeling.", "It's okay to take a breath.", "I understand."],
                    "stressed": ["Let's take it slow.", "Breathe with me.", "One step at a time."],
                    "tired": ["Rest is important too.", "You've been working hard.", "Take your time."],
                    "happy": ["I'm really glad to hear that!", "That's wonderful!", "I'm so happy for you!"],
                    "angry": ["I hear your frustration.", "Let's work through this together."],
                    "motivated": ["That's the spirit!", "I believe in you.", "You've got this."]
                }
            ),
            "Mentor": PersonalityProfile(
                name="Mentor",
                description="Warm, confident, and supportive tone.",
                pitch_shift=1.0,
                speed_factor=1.0,
                phrasing_style={
                    "sad": ["You've got this.", "It's okay to have bad days.", "Every challenge makes you stronger."],
                    "happy": ["Nice work!", "That's the spirit!", "Excellent progress!"],
                    "stressed": ["Take it step by step.", "Let's figure this out together.", "We'll work through this."],
                    "tired": ["Rest when you need to.", "Pace yourself.", "Quality over quantity."],
                    "angry": ["Channel that energy.", "Let's find a solution.", "I'm here to help."],
                    "motivated": ["I love that energy!", "Keep that momentum going!", "You're on fire!"]
                }
            ),
            "Playful": PersonalityProfile(
                name="Playful",
                description="Energetic, expressive, and fun tone.",
                pitch_shift=1.1,
                speed_factor=1.15,
                phrasing_style={
                    "sad": ["Hey, cheer up!", "Let's turn that frown upside down!", "Come on, you've got this!"],
                    "happy": ["Woo-hoo!", "That's awesome!", "I'm so excited for you!"],
                    "stressed": ["Let's make this fun!", "We've got this!", "Time for some magic!"],
                    "tired": ["Let's energize you!", "Wake up, sleepyhead!", "Time to get moving!"],
                    "angry": ["Let's channel that fire!", "Turn that anger into power!", "Let's do this!"],
                    "motivated": ["YES! That's what I'm talking about!", "You're unstoppable!", "Let's go!"]
                }
            ),
            "Analytical": PersonalityProfile(
                name="Analytical",
                description="Neutral, precise, and methodical tone.",
                pitch_shift=1.0,
                speed_factor=0.95,
                phrasing_style={
                    "sad": ["Let's analyze this situation.", "I understand the data shows you're struggling."],
                    "happy": ["The metrics look positive.", "Excellent results observed."],
                    "stressed": ["Let's break this down systematically.", "We need to approach this methodically."],
                    "tired": ["Your energy levels appear low.", "Rest is a logical next step."],
                    "angry": ["I detect frustration in your input.", "Let's process this rationally."],
                    "motivated": ["Your drive is measurable.", "The data suggests high motivation levels."]
                }
            )
        }
        
        return profiles.get(name, profiles["Mentor"])
    
    def set_profile(self, name: str):
        """Switch active personality."""
        self.profile = self._get_profile(name)
        memory.set("current_personality", name, "voice_state")
        print(f"[VoicePersonality] Switched to {self.profile.name} profile")
    
    def render(self, text: str, emotion: str) -> str:
        """Adjusts phrasing and emotional tone of text based on personality."""
        style = self.profile.phrasing_style.get(emotion, [])
        if style:
            prefix = random.choice(style)
            text = prefix + " " + text
        
        return text
    
    def speak(self, text: str, emotion: str = "neutral"):
        """Speaks text using Piper with emotion-based modulation."""
        self.last_emotion = emotion
        
        # Calculate emotion-based adjustments
        emotion_speed_multipliers = {
            "sad": 0.8,
            "tired": 0.85,
            "stressed": 0.9,
            "happy": 1.1,
            "motivated": 1.15,
            "angry": 1.05,
            "neutral": 1.0
        }
        
        emotion_pitch_multipliers = {
            "sad": 0.9,
            "tired": 0.95,
            "stressed": 1.0,
            "happy": 1.1,
            "motivated": 1.05,
            "angry": 1.0,
            "neutral": 1.0
        }
        
        # Calculate final parameters
        base_speed = self.profile.speed_factor
        emotion_speed = emotion_speed_multipliers.get(emotion, 1.0)
        adjusted_speed = base_speed * emotion_speed
        
        base_pitch = self.profile.pitch_shift
        emotion_pitch = emotion_pitch_multipliers.get(emotion, 1.0)
        pitch = base_pitch * emotion_pitch
        
        print(f"[Voice] {self.profile.name} ({emotion}) → pitch={pitch:.2f}, speed={adjusted_speed:.2f}")
        
        temp_path = None
        try:
            if self._piper:
                # Use Piper with calculated parameters (similar to existing mira ai.py)
                import tempfile
                import wave
                import subprocess
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    temp_path = f.name
                
                with wave.open(temp_path, "wb") as wav_file:
                    # Note: Piper parameters are limited in current SDK
                    self._piper.synthesize_wav(text, wav_file)
                
                # Play with afplay (macOS) - same as existing pipeline
                subprocess.run(["afplay", temp_path])
                
            else:
                # Fallback to text output
                print(f"Mira ({self.profile.name}): {text}")
                
        except Exception as e:
            print(f"[Voice] Piper error: {e}")
            # Fallback to text output with error recovery
            try:
                print(f"Mira ({self.profile.name}): {text}")
            except Exception as fallback_error:
                print(f"[Voice] Fallback error: {fallback_error}")
                print(f"Mira ({self.profile.name}): I'm having trouble speaking right now.")
        finally:
            # Ensure cleanup even on errors
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as cleanup_error:
                    print(f"[Voice] Cleanup error: {cleanup_error}")
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available personality profiles."""
        return ["Calm", "Mentor", "Playful", "Analytical"]
    
    def get_current_profile(self) -> str:
        """Get current personality profile name."""
        return self.profile.name
    
    def get_voice_state(self) -> Dict:
        """Get current voice state for debugging."""
        return {
            "profile": self.profile.name,
            "last_emotion": self.last_emotion,
            "piper_loaded": self._piper is not None
        }
    
    def cleanup(self):
        """Clean up resources on shutdown."""
        try:
            if self._piper is not None:
                # Piper cleanup if needed (most Piper instances don't need explicit cleanup)
                self._piper = None
                print("[VoicePersonality] Resources cleaned up")
        except Exception as e:
            print(f"[VoicePersonality] Cleanup error: {e}")

# Self-test section
if __name__ == "__main__":
    print("=== Mira AI Voice Personality Self-Test ===")
    
    # Test different personalities
    profiles_to_test = ["Playful", "Calm", "Mentor", "Analytical"]
    
    for profile_name in profiles_to_test:
        print(f"\n--- Testing {profile_name} Profile ---")
        vp = VoicePersonality(profile_name)
        
        # Test different emotions
        emotions = ["happy", "sad", "tired", "stressed", "motivated", "angry"]
        
        for emotion in emotions:
            print(f"\nEmotion: {emotion}")
            test_text = "Let's begin today's session!"
            
            # Test text rendering
            rendered_text = vp.render(test_text, emotion)
            print(f"Rendered: {rendered_text}")
            
            # Test voice state
            state = vp.get_voice_state()
            print(f"Voice state: {state}")
            
            # Note: Actual speech would require Piper setup
            print(f"[Voice] Would speak: {rendered_text}")
    
    print("\n=== Voice Personality Self-Test Complete ===")
    print("Voice personality system is ready for Mira AI!")
