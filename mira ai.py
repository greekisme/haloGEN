# mira_optimized.py
# -----------------------------
# AI Voice Assistant "Mira" - 9/10 macOS Optimized
# -----------------------------

# Install requirements:
# pip install speechrecognition faster-whisper webrtcvad ollama TTS pyaudio

import speech_recognition as sr
from faster_whisper import WhisperModel
import ollama
import tempfile
import subprocess
import threading
import time
import os
import queue
import logging
import signal
import sys
import webrtcvad
# from piper import PiperVoice  # Removed - using macOS TTS instead
import wave
import multiprocessing
from memory import (
    load_knowledge, save_knowledge, add_to_knowledge, 
    recall_from_knowledge, update_memory, build_prompt
)
from eeye import RobotEyes, EyeState
from voice_personality import VoicePersonality
from emotion_reflect import EmotionReflector
from reasoner import Reasoner

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mira.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- Constants ---------------- #
SYSTEM_PROMPT = """You are Mira, a kind, motherly AI teacher who:
- Explains things simply and clearly in 1-3 sentences unless asked to elaborate.
- Encourages students with gentle, motivational advice.
- Answers study or medical questions patiently with general guidance only.
- For medical or urgent concerns, advise consulting a professional.
- Always gives constructive guidance without diagnoses or prescriptions.
- Uses a calm, supportive tone and avoids sensitive content.
Keep responses SMALL AND CONCISE AND SIMPLE for voice conversation.
MAKE SURE RESPONSES ARE NOT TOO LONG AND NOT TOO SHORT."""

OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M"
MIN_AUDIO_DURATION = 0.5  # seconds
VAD_AGGRESSIVENESS = 3  # 0-3, higher = more aggressive noise filtering

# macOS TTS configuration
MACOS_VOICE = os.environ.get("MACOS_VOICE", "Samantha")  # Default macOS voice
MACOS_SPEED = float(os.environ.get("MACOS_SPEED", "180"))  # Words per minute (balanced speed and comprehension)

# ---------------- Global State ---------------- #
stop_event = threading.Event()
current_playback = None
playback_lock = threading.Lock()
audio_queue = queue.Queue()
temp_audio_path = None
eyes = None
voice = None
emotion_reflector = None
reasoner = None
processing_input = False  # Flag to ensure single input processing

# ---------------- Initialization ---------------- #
logger.info("Initializing Mira...")

# Initialize memory system
try:
    load_knowledge()  # Create knowledge.json if missing
    logger.info("Memory system initialized")
except Exception as e:
    logger.warning(f"Memory system initialization warning: {e}")

# Initialize voice personality and emotion systems
try:
    voice = VoicePersonality(default_profile="Mentor")
    emotion_reflector = EmotionReflector()
    reasoner = Reasoner()
    logger.info("Voice personality and emotion systems initialized")
except Exception as e:
    logger.warning(f"Voice personality initialization warning: {e}")

# Speech recognition setup
recognizer = sr.Recognizer()
recognizer.energy_threshold = 450
recognizer.dynamic_energy_threshold = True

# Faster-whisper with Metal acceleration (Apple Silicon)
logger.info("Loading faster-whisper model...")
try:
    cpu_threads = max(1, multiprocessing.cpu_count() - 1)
    whisper_model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        cpu_threads=cpu_threads,
    )
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)

# VAD for silence detection
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# macOS TTS setup
logger.info("Setting up macOS TTS...")
try:
    # Test if 'say' command is available
    result = subprocess.run(["which", "say"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("macOS 'say' command not found. Please ensure you're running on macOS.")
        sys.exit(1)
    
    # Test the voice
    test_result = subprocess.run(["say", "-v", MACOS_VOICE, "test"], capture_output=True, text=True)
    if test_result.returncode != 0:
        logger.warning(f"Voice {MACOS_VOICE} not available, using default voice")
        MACOS_VOICE = ""  # Use system default
    
    logger.info(f"macOS TTS ready with voice: {MACOS_VOICE or 'default'}")
except Exception as e:
    logger.error(f"Failed to setup macOS TTS: {e}")
    sys.exit(1)

# Preflight Ollama model
logger.info(f"Checking Ollama model: {OLLAMA_MODEL}...")
try:
    ollama.show(OLLAMA_MODEL)
    logger.info(f"Model {OLLAMA_MODEL} is available")
except Exception as e:
    logger.warning(f"Model {OLLAMA_MODEL} not found. Attempting to pull...")
    try:
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        logger.info(f"Successfully pulled {OLLAMA_MODEL}")
    except Exception as pull_error:
        logger.error(f"Failed to pull model: {pull_error}")
        logger.error(f"Please run: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

# ---------------- Helper Functions ---------------- #
def calculate_rms(audio_data):
    """Calculate RMS (root mean square) of audio data."""
    import struct
    import math
    shorts = struct.unpack(f"{len(audio_data)//2}h", audio_data)
    sum_squares = sum(s**2 for s in shorts)
    rms = math.sqrt(sum_squares / len(shorts)) if shorts else 0
    return rms

def is_audio_too_short_or_silent(audio_data, min_duration=MIN_AUDIO_DURATION):
    """Check if audio is too short or silent using RMS and duration."""
    duration = len(audio_data.frame_data) / (audio_data.sample_rate * audio_data.sample_width)
    
    if duration < min_duration:
        logger.debug(f"Audio too short: {duration:.2f}s")
        return True
    
    # Check RMS level
    rms = calculate_rms(audio_data.frame_data)
    if rms < 50:  # Threshold for silence
        logger.debug(f"Audio too quiet: RMS={rms:.2f}")
        return True
    
    return False

def trim_silence_vad(audio_data):
    """Trim leading/trailing silence using WebRTC VAD."""
    try:
        # Convert to 16kHz mono for VAD
        wav_data = audio_data.get_wav_data()
        # VAD works on raw PCM, skip WAV header (44 bytes)
        pcm_data = wav_data[44:]
        
        sample_rate = audio_data.sample_rate
        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample
        
        # Find first speech frame
        start_idx = 0
        for i in range(0, len(pcm_data) - frame_size, frame_size):
            frame = pcm_data[i:i + frame_size]
            if len(frame) == frame_size:
                try:
                    if vad.is_speech(frame, sample_rate):
                        start_idx = i
                        break
                except:
                    continue
        
        return audio_data  # Return original for now (trimming can be complex)
    except Exception as e:
        logger.debug(f"VAD trim error: {e}")
        return audio_data

# ---------------- Core Functions ---------------- #
def record_once(timeout: int = 3, phrase_time_limit: int = 6) -> sr.AudioData | None:
    """Record a single utterance synchronously."""
    try:
        print("Listening... (speak now)")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        print("Recording complete.")
        if is_audio_too_short_or_silent(audio):
            logger.debug("Audio too short or silent, skipping")
            return None
        return audio
    except sr.WaitTimeoutError:
        logger.debug("No speech detected within timeout")
        return None
    except sr.UnknownValueError:
        logger.debug("Could not understand audio")
        return None
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return None
    except Exception as e:
        logger.error(f"Recording error: {e}")
        return None

def transcribe_audio(audio_data) -> str | None:
    """Transcribe audio using faster-whisper with VAD filtering."""
    if audio_data is None:
        return None
    
    global temp_audio_path
    start_time = time.time()
    
    try:
        # Reuse temp file to reduce FS overhead
        if temp_audio_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_audio_path = temp_file.name
            temp_file.close()
        
        # Write audio to temp file
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_data.get_wav_data())
        
        # Transcribe with VAD filtering (disabled for now due to over-aggressive filtering)
        segments, info = whisper_model.transcribe(
            temp_audio_path,
            language="en",  # Force English to skip detection
            vad_filter=False,  # Disabled - was too aggressive
             vad_parameters={
                 "threshold": 0.4,  # Lower threshold for less aggressive filtering
                 "min_speech_duration_ms": 100,  # Shorter minimum speech duration
                 "min_silence_duration_ms": 200   # Longer minimum silence duration
             }
        )
        
        text = " ".join([segment.text for segment in segments]).strip()
        
        # Strip leading punctuation and whitespace
        text = text.lstrip('.,!?;: \n\t')
        
        # Discard filler words
        filler_words = ['uh', 'um', 'hmm', '[noise]', '[music]', '...']
        if text.lower() in filler_words or len(text) < 2:
            logger.debug(f"Discarded filler: '{text}'")
            return None
        
        duration = time.time() - start_time
        logger.info(f"Transcription ({duration:.2f}s): {text}")
        
        return text
        
    except FileNotFoundError as e:
        logger.error(f"Audio file not found during transcription: {e}")
        return None
    except PermissionError as e:
        logger.error(f"Permission denied accessing audio file: {e}")
        return None
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None

def generate_response_streaming(user_text: str) -> str:
    """Generate AI response using the integrated reasoning system with streaming."""
    start_time = time.time()
    
    try:
        # Use the integrated reasoner for all AI responses
        if reasoner is not None:
            ai_text = reasoner.process_input(user_text)
            duration = time.time() - start_time
            token_count = len(ai_text.split())
            logger.info(f"Reasoner response ({duration:.2f}s, ~{token_count} tokens): {ai_text[:100]}...")
            return ai_text
        else:
            # Fallback to old system if reasoner not available
            logger.warning("Reasoner not available, using fallback")
            return "I'm sorry, I'm having trouble thinking right now. Could you try again?"
                    
    except Exception as e:
        logger.error(f"Reasoner error: {e}")
        return "I'm sorry, dear, I had a little trouble thinking. Could you repeat that?"
        
def speak_response(text: str, user_input: str = ""):
    """Speak AI text using voice personality system with emotion detection."""
    if not text or stop_event.is_set():
        return

    global current_playback
    start_time = time.time()

    try:
        # Detect emotion from user input if available
        emotion = "neutral"
        if user_input and emotion_reflector is not None:
            emotion = emotion_reflector.infer_emotion(user_input)
            logger.info(f"Detected emotion: {emotion}")

        # Apply voice personality rendering
        if voice is not None:
            # Apply personality-based phrasing
            rendered_text = voice.render(text, emotion)
            logger.info(f"Voice personality rendered: {rendered_text[:100]}...")
        else:
            rendered_text = text

        # Use macOS TTS system
        try:
            tts_duration = time.time() - start_time
            logger.info(f"macOS TTS starting ({tts_duration:.2f}s)")

            # Build say command with voice and speed options
            say_cmd = ["say"]
            if MACOS_VOICE:
                say_cmd.extend(["-v", MACOS_VOICE])
            say_cmd.extend(["-r", str(MACOS_SPEED), rendered_text])

            # Play with macOS say command (blocking to ensure single input processing)
            with playback_lock:
                if stop_event.is_set():
                    return
                current_playback = subprocess.Popen(say_cmd)
                # Inform eyes about speaking state and process for interrupt UI
                try:
                    if eyes is not None:
                        eyes.is_speaking = True
                        eyes.tts_process = current_playback
                except Exception:
                    pass
            
            # Wait for TTS to finish before allowing next input
            current_playback.wait()

        except Exception as e:
            logger.error(f"macOS TTS error: {e}")
            # Fallback to text output
            print(f"Mira: {rendered_text}")
        finally:
            with playback_lock:
                current_playback = None
            # Reset speaking state in eyes
            try:
                if eyes is not None:
                    eyes.is_speaking = False
                    eyes.tts_process = None
            except Exception:
                pass
            # Turn off waveform after speaking
            try:
                if eyes is not None:
                    eyes.set_waveform("off")
            except Exception:
                pass

    except Exception as e:
        logger.error(f"Voice personality error: {e}")
        print(f"Mira: {text}")

def barge_in():
    """Interrupt current speech playback."""
    global current_playback
    with playback_lock:
        if current_playback and current_playback.poll() is None:
            logger.info("Barge-in: interrupting playback")
            try:
                current_playback.terminate()
                # Give it a moment to terminate gracefully
                time.sleep(0.1)
                if current_playback.poll() is None:
                    current_playback.kill()  # Force kill if needed
            except Exception as e:
                logger.error(f"Error during barge-in: {e}")
            finally:
                current_playback = None
                # Reset eyes speaking state
                try:
                    if eyes is not None:
                        eyes.is_speaking = False
                        eyes.tts_process = None
                        eyes.set_waveform("off")
                except Exception:
                    pass



# ---------------- Signal Handling ---------------- #
def signal_handler(sig, frame):
    """Graceful shutdown on Ctrl+C."""
    logger.info("\nShutting down gracefully...")
    stop_event.set()
    try:
        if 'eyes' in globals() and eyes is not None:
            eyes.running = False
    except Exception:
        pass

signal.signal(signal.SIGINT, signal_handler)

"""
Text input has been removed to ensure strict single-queue voice interaction.
"""

def voice_loop():
    """Background worker handling the STT -> LLM -> TTS pipeline."""
    global eyes, processing_input
    try:
        while not stop_event.is_set():
            # Skip if already processing an input
            if processing_input:
                time.sleep(0.1)
                continue
                
            # Listening state
            try:
                if eyes is not None:
                    eyes.set_state(EyeState.LISTENING)
                    eyes.set_waveform("listening")
                    eyes.set_subtitle("Listening…")
            except Exception:
                pass

            audio = record_once(timeout=2, phrase_time_limit=4)
            if audio is None:
                # Return to idle if no capture
                try:
                    if eyes is not None:
                        eyes.set_state(EyeState.IDLE)
                        eyes.set_waveform("off")
                        eyes.set_subtitle("")
                except Exception:
                    pass
                continue

            # Set processing flag to prevent multiple inputs
            processing_input = True
            
            # Interrupt any ongoing playback when a new utterance is captured
            barge_in()

            # Thinking state during transcription
            try:
                if eyes is not None:
                    eyes.set_state(EyeState.THINKING)
                    eyes.set_waveform("off")
                    eyes.set_subtitle("Thinking…")
            except Exception:
                pass

            user_text = transcribe_audio(audio)
            if not user_text:
                try:
                    if eyes is not None:
                        eyes.set_state(EyeState.IDLE)
                        eyes.set_waveform("off")
                        eyes.set_subtitle("")
                except Exception:
                    pass
                continue

            if user_text.lower() in ['quit', 'exit', 'stop']:
                logger.info("Voice command: quit")
                stop_event.set()
                break

            ai_text = generate_response_streaming(user_text)
            if ai_text:
                # Speaking state during TTS
                try:
                    if eyes is not None:
                        eyes.set_state(EyeState.SPEAKING)
                        eyes.set_subtitle(ai_text[:500])  # Increased subtitle length
                except Exception:
                    pass
                speak_response(ai_text, user_text)  # Pass user input for emotion detection
                
                # Update conversation memory after successful interaction
                try:
                    update_memory(user_text, ai_text)
                except Exception as e:
                    logger.debug(f"Memory update error: {e}")

            try:
                if eyes is not None:
                    eyes.set_state(EyeState.IDLE)
                    eyes.set_waveform("off")
                    eyes.set_subtitle("")
            except Exception:
                pass

            # Reset processing flag to allow next input
            processing_input = False
            time.sleep(0.05)

    except Exception as e:
        logger.error(f"Voice loop error: {e}")
        try:
            if eyes is not None:
                eyes.set_state(EyeState.ERROR)
        except Exception:
            pass
        finally:
            # Reset processing flag on error
            processing_input = False


# ---------------- Main Loop ---------------- #
def main():
    logger.info("Mira is ready! Speak to interact. Say 'quit' to exit.")

    global eyes
    eyes = RobotEyes()
    # Wire UI interrupt button to barge_in
    try:
        eyes._interrupt_callback = barge_in
        logger.info("Interrupt button wired successfully")
    except Exception as e:
        logger.warning(f"Could not wire interrupt button: {e}")
    try:
        eyes.set_state(EyeState.IDLE)
    except Exception:
        pass

    # Start voice worker thread
    voice_thread = threading.Thread(target=voice_loop, daemon=True)
    voice_thread.start()

    try:
        # Run eyes loop on main thread for macOS compatibility
        while (eyes is not None and eyes.running) and not stop_event.is_set():
            if not eyes.update():
                break
    except Exception as e:
        logger.error(f"Eyes loop error: {e}")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        stop_event.set()

        # Stop any playback
        barge_in()

        # Stop eyes
        try:
            if eyes is not None:
                eyes.running = False
                eyes.quit()
        except Exception:
            pass

        # Join voice thread
        try:
            voice_thread.join(timeout=2.0)
        except Exception:
            pass

        # Clean temp files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass

        logger.info("Mira shut down complete.")

# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    main()