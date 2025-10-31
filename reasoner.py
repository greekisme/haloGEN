# Mira AI Reasoner
# Connects memory, symbolic logic, and LLM for hybrid neurosymbolic reasoning.
# Enables Mira to think, infer, and respond intelligently.

import time
import os
import re
import requests
import json
import logging
from functools import lru_cache
from typing import Optional
from memory import memory, update_memory, build_prompt
from logic_engine import rule_engine, infer
from emotion_reflect import EmotionReflector

MAX_CACHE_SIZE = 1000

# Ensure global logging is configured:
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Ollama persistent session setup ---
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_HEADERS = {"Connection": "keep-alive"}
_ollama_session = requests.Session()

class Reasoner:
    def __init__(self, model: str = "llama3:8b-instruct-q4_K_M"):
        self.model = model
        self.emotion_reflector = EmotionReflector()
        
        # Enhanced truth base for fact validation
        self.FACT_BASE = {
            # Celestial bodies
            "sun": "star", "earth": "planet", "moon": "satellite", "mars": "planet",
            "jupiter": "planet", "saturn": "planet", "venus": "planet", "mercury": "planet",
            
            # States of matter
            "water": "liquid", "ice": "solid", "steam": "gas", "fire": "plasma",
            "air": "gas", "rock": "solid", "metal": "solid", "wood": "solid",
            
            # Physical forces
            "gravity": "force", "magnetism": "force field", "electricity": "flow of electrons",
            "light": "electromagnetic radiation", "sound": "mechanical wave", "heat": "thermal energy",
            
            # Basic science
            "atoms": "smallest unit of matter", "molecules": "groups of atoms", "cells": "basic unit of life",
            "dna": "genetic material", "proteins": "biological molecules", "enzymes": "biological catalysts",
            
            # Biological processes
            "photosynthesis": "process plants use to make food", "respiration": "process of breathing",
            "digestion": "process of breaking down food", "circulation": "movement of blood through body",
            "metabolism": "chemical processes in living organisms", "reproduction": "process of creating offspring",
            
            # Colors
            "red": "color", "blue": "color", "green": "color", "yellow": "color", "black": "color", "white": "color",
            
            # Animals
            "dog": "domesticated mammal", "cat": "domesticated feline", "bird": "flying animal",
            "fish": "aquatic animal", "horse": "domesticated mammal", "cow": "domesticated mammal",
            
            # Plants
            "tree": "woody plant", "flower": "reproductive structure", "grass": "ground cover plant",
            "fruit": "ripened ovary", "seed": "plant embryo", "leaf": "plant organ for photosynthesis",
            
            # Weather
            "rain": "precipitation", "snow": "frozen precipitation", "wind": "movement of air",
            "cloud": "visible mass of water droplets", "storm": "severe weather", "sunshine": "direct sunlight"
        }
        
        # Performance monitoring
        self._performance_stats = {
            'memory_operations': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0
        }
        
        # Cache for expensive operations
        self._semantic_cache = {}
        self._classification_cache = {}
        
        logger.info("[Reasoner] Initialized with emotion reflection capabilities, truth validation, and performance monitoring")

    def classify_statement(self, text: str) -> str:
        """Enhanced classification with better educational query detection."""
        text = text.lower().strip()
        
        # Check cache first for performance
        if text in self._classification_cache:
            self._performance_stats['cache_hits'] += 1
            return self._classification_cache[text]
        
        self._performance_stats['cache_misses'] += 1
        
        # Enhanced educational/question patterns
        educational_patterns = [
            r"^(who|what|when|where|why|how)\s+",
            r"^(can you|could you|would you)\s+",
            r"^(explain|describe|define|tell me about)\s+",
            r"^what is (the|a|an)?\s+",
            r"^how does (the|a|an)?\s+",
            r"^why is (the|a|an)?\s+",
            r"^when was (the|a|an)?\s+",
            r"^where is (the|a|an)?\s+",
            r"difference between",
            r"types? of",
            r"how to",
            r"what are",
            r"how do",
            r"which one",
            r"why do",
            r"how do",
            r"what causes",
            r"what happens",
            r"differentiate between"
        ]
        
        # Check for educational questions first
        if any(re.search(p, text) for p in educational_patterns):
            # Exclude personal questions from educational classification
            personal_indicators = ['my', 'i ', 'me', 'mine', 'myself']
            if not any(indicator in text for indicator in personal_indicators):
                result = "educational_query"
                self._classification_cache[text] = result
                return result
        
        # Enhanced identity patterns using regex for precise matching
        IDENTITY_PATTERNS = [
            r"\bmy name is\b",
            r"\bi am called\b", 
            r"\bi (live|work|study|am from|was born) in\b",
            r"\bi was born\b",
            r"\bi am from\b",
            r"\bi['']m\b",
            r"\bi am a\b",
            r"\bi am an\b",
            r"\bi am the\b",
            r"\bmy age is\b",
            r"\bi am \d+\b",
            r"\bmy job is\b",
            r"\bi work as\b",
            r"\bi study at\b",
            r"\bmy profession is\b",
            r"\bi'm a\b",
            r"\bi'm an\b",
            r"\bmy occupation is\b",
            r"\bi reside in\b",
            r"\bmy hometown is\b",
            r"\bi come from\b",
            r"\bmy nationality is\b",
            r"\bi am \d+ years old\b",
            r"\bmy birthday is\b"
        ]
        
        if any(re.search(p, text) for p in IDENTITY_PATTERNS):
            result = "user_identity"
        else:
            # Enhanced personal preference indicators
            preference_indicators = [
                "i like", "i love", "i enjoy", "i prefer", "i hate", "i dislike",
                "my favorite", "i want", "i need", "i wish", "i hope",
                "i think", "i believe", "i feel", "i want to", "i would like"
            ]
            
            if any(p in text for p in preference_indicators):
                result = "personal_preference"
            else:
                # Check for conversation/chat statements
                conversation_indicators = [
                    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                    "how are you", "what's up", "how's it going", "nice to meet you",
                    "thank you", "thanks", "you're welcome", "goodbye", "bye", "see you",
                    "have a good day", "take care", "talk to you later", "catch you later"
                ]
                
                if any(indicator in text for indicator in conversation_indicators):
                    result = "conversation"
                else:
                    result = "general_query"
        
        # Cache the result for future use
        self._classification_cache[text] = result
        return result

    def validate_fact(self, statement: str) -> tuple[bool, str]:
        """Validate a factual claim against the truth base. Returns (is_valid, correction)."""
        text = statement.lower().strip()
        
        # Enhanced parsing for subject-predicate structure
        m = re.match(r"^([\w\s]+?)\s+is\s+(?:a|an|the)?\s*([\w\s]+)$", text)
        if not m:
            return False, "Unable to parse as factual statement."
        
        subject, predicate = m.groups()
        subject, predicate = subject.strip(), predicate.strip()
        
        # Remove articles for better matching
        subject_clean = re.sub(r'^(a|an|the)\s+', '', subject)
        predicate_clean = re.sub(r'^(a|an|the)\s+', '', predicate)
        
        # Check against enhanced fact base
        if subject_clean in self.FACT_BASE:
            true_value = self.FACT_BASE[subject_clean]
            if predicate_clean == true_value or predicate_clean in true_value:
                return True, f"Confirmed: {subject} is a {true_value}."
            else:
                return False, f"Actually, {subject} is a {true_value}."
        
        # Check for common factual patterns
        if self._is_common_factual_pattern(subject_clean, predicate_clean):
            return True, f"Confirmed: {subject} is {predicate}."
        elif self._is_known_false_pattern(subject_clean, predicate_clean):
            return False, f"That's not correct. {subject} is not {predicate}."
        
        # Unrecognized subject → treat as unverified
        return False, "I can't verify this; saving as a belief for now."
    
    def _is_common_factual_pattern(self, subject: str, predicate: str) -> bool:
        """Check if this matches common factual patterns."""
        common_patterns = [
            ("water", "wet"), ("fire", "hot"), ("ice", "cold"), ("sun", "bright"),
            ("moon", "round"), ("earth", "round"), ("sky", "blue"), ("grass", "green"),
            ("snow", "white"), ("night", "dark"), ("day", "bright"), ("ocean", "blue")
        ]
        return (subject, predicate) in common_patterns
    
    def _is_known_false_pattern(self, subject: str, predicate: str) -> bool:
        """Check if this matches known false patterns."""
        false_patterns = [
            ("earth", "flat"), ("sun", "cold"), ("water", "dry"), ("fire", "cold"),
            ("ice", "hot"), ("moon", "square"), ("sky", "red"), ("grass", "red")
        ]
        return (subject, predicate) in false_patterns

    def process_input(self, user_input: str) -> str:
        """Process user input through the complete reasoning pipeline."""
        start_time = time.time()
        logger.info(f"[Reasoner] User input received: {user_input}")
        
        # Performance monitoring
        self._performance_stats['memory_operations'] += 1

        # Step 1: Check for cached responses first (ChatGPT-like caching)
        cached_response = self._get_cached_response(user_input)
        if cached_response:
            logger.info("[Reasoner] Using cached response for faster reply")
            return cached_response

        # Step 2: ChatGPT-like behavior - always include conversational context
        memory_context = self._build_conversational_context()
        if memory_context:
            logger.info(f"[Reasoner] Conversation context: {memory_context[:100]}...")
        else:
            logger.info("[Reasoner] No recent conversation context")

        # Step 3: Logic Inference
        user_state = self._extract_user_state(user_input)
        
        context = {
            "user": {
                "input": user_input, 
                "skip_study": user_state.get("skip_study", False),
                "tired": user_state.get("tired", False),
                "mood": user_state.get("mood", "neutral")
            },
            "ai_response": "",
            "timestamp": time.time()
        }

        context = infer(context)
        logger.info("[Reasoner] Logic inference complete.")

        # Step 4: Emotional Adjustment
        emotion = self.emotion_reflector.infer_emotion(user_input)
        logger.info(f"[Reasoner] Inferred emotion: {emotion}")

        # Step 5: LLM Response (if no rule-based response)
        if context.get("ai_response"):
            final_response = context["ai_response"]
            logger.info("[Reasoner] Using rule-based response")
        else:
            # Use unified LLM method with optional memory context
            final_response = self._generate_llm_response_fast(user_input, memory_context)
            if memory_context:
                logger.info("[Reasoner] Using LLM-generated response with memory context")
            else:
                logger.info("[Reasoner] Using LLM-generated response without memory context")

        # Step 6: Apply emotional tone adjustment
        final_response = self.emotion_reflector.adjust_tone(final_response, emotion)
        
        # Step 7: Memory Update with Validation
        self._update_memory_with_validation(user_input, final_response)
        
        # Performance monitoring
        processing_time = time.time() - start_time
        self._performance_stats['total_processing_time'] += processing_time
        if processing_time > 1.0:  # Log slow operations
            logger.warning(f"[Reasoner] Slow processing: {processing_time:.2f}s")
        
        logger.info("[Reasoner] Response generated and saved to memory.")
        
        return final_response

    def _get_cached_response(self, user_input: str) -> Optional[str]:
        """ChatGPT-like caching for common responses."""
        user_input_lower = user_input.lower().strip()
        
        # Enhanced quick responses cache
        quick_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What can I do for you?",
            "hey": "Hey! How can I assist you?",
            "how are you": "I'm doing well, thank you! How can I help you today?",
            "what's up": "Not much! I'm here to help with any questions you have.",
            "good morning": "Good morning! How can I assist you today?",
            "good afternoon": "Good afternoon! What can I help you with?",
            "good evening": "Good evening! How can I be of service?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Feel free to ask if you need anything else.",
            "bye": "Goodbye! Have a great day!",
            "goodbye": "Goodbye! Take care!",
            "see you later": "See you later! Have a wonderful day!",
            "what can you do": "I can help answer questions, explain concepts, assist with problem-solving, and have conversations on a wide variety of topics. What would you like to know?",
            "who are you": "I'm Mira, an AI assistant designed to help answer questions and have conversations. How can I assist you today?",
            "what is your name": "My name is Mira! I'm here to help you with any questions or tasks you might have.",
            "help": "I'm here to help! You can ask me questions about various topics, request explanations, or just have a conversation. What would you like to know?",
            "okay": "Great! What would you like to do next?",
            "ok": "Perfect! How can I help you?",
            "yes": "Excellent! What would you like to know?",
            "no": "No problem! Is there anything else I can help you with?",
            "sure": "Wonderful! What can I assist you with?",
            "maybe": "That's fine! Let me know if you'd like to explore anything specific.",
            "i don't know": "That's okay! I'm here to help you figure things out. What would you like to learn about?",
            "i'm not sure": "No worries! I can help you find answers. What topic interests you?",
            "explain": "I'd be happy to explain! What would you like me to clarify?",
            "tell me about": "I'd be glad to tell you about that! What specific topic are you interested in?",
            "what is": "I can explain that for you! What would you like to know about?",
            "how does": "I can explain how that works! What process are you curious about?",
            "why": "Great question! What would you like me to explain the reasoning behind?",
            "when": "I can help with timing information! What event or process are you asking about?",
            "where": "I can help with location information! What place are you curious about?",
            "who": "I can provide information about people! Who would you like to know about?",
            "which": "I can help you choose! What options are you considering?"
        }
        
        # Check for exact matches first
        if user_input_lower in quick_responses:
            logger.info("[Reasoner] Using cached quick response")
            return quick_responses[user_input_lower]
        
        # Check for partial matches (very simple patterns only)
        for key, response in quick_responses.items():
            if key in user_input_lower and len(user_input_lower) < 20:  # Only for very short inputs
                logger.info(f"[Reasoner] Using cached response for '{key}'")
                return response
        
        return None

    def _build_conversational_context(self) -> str:
        """Build simple conversational context like ChatGPT - only recent conversation."""
        try:
            conversation_context = self._get_conversation_context("")
            if conversation_context and len(conversation_context.strip()) > 0:
                # Limit to last few exchanges for context
                if len(conversation_context) > 300:
                    conversation_context = conversation_context[-297:] + "..."
                return f"Recent conversation: {conversation_context}"
        except Exception as e:
            logger.warning(f"[Reasoner] Failed to get conversation context: {e}")
        return ""

    def _update_memory_with_validation(self, user_input: str, ai_response: str):
        """Simplified memory update - only store conversation, no individual facts."""
        # Only store conversation exchange, no individual fact storage
        logger.info("[Reasoner] Storing conversation exchange only (no individual facts)")
        
        # Always store the conversation exchange
        update_memory(user_input, ai_response)
        
        # Skip memory cleanup to avoid errors
        logger.info("[Reasoner] Skipping memory cleanup for stability")

    def _perform_memory_cleanup(self):
        """More aggressive memory cleanup."""
        try:
            from memory import memory
            
            # Get memory statistics
            stats = memory.get_memory_stats()
            
            # Clean up expired entries
            memory.cleanup_expired_entries()
            
            # More aggressive conversation cleanup
            if stats['conversation_entries'] > 15:
                memory.cleanup_old_conversations(max_conversations=8)
            
            # More aggressive low confidence cleanup
            if stats['low_confidence_entries'] > 30:
                memory.cleanup_low_confidence_entries(min_confidence=0.3)
            
            # Clean up old queries
            memory.cleanup_old_entries(max_age_hours=24)
            
            logger.info(f"[Reasoner] Aggressive memory cleanup completed")
            
        except Exception as e:
            logger.error(f"[Reasoner] Memory cleanup error: {e}")

    def _extract_user_state(self, user_input: str) -> dict:
        """Extract user state from input for logic engine."""
        user_input_lower = user_input.lower()
        
        state = {
            "skip_study": False,
            "tired": False,
            "mood": "neutral"
        }
        
        # Check for study-related intentions
        skip_study_phrases = [
            "skip studying", "skip study", "not studying", "not study",
            "skip today", "skip my studies", "skip my study"
        ]
        if any(phrase in user_input_lower for phrase in skip_study_phrases):
            state["skip_study"] = True
        
        # Check for tiredness indicators
        tired_phrases = [
            "tired", "exhausted", "fatigued", "sleepy", "drained",
            "worn out", "beat", "wiped out", "really tired"
        ]
        if any(phrase in user_input_lower for phrase in tired_phrases):
            state["tired"] = True
        
        # Check for mood indicators
        if any(word in user_input_lower for word in ["sad", "depressed", "down", "blue"]):
            state["mood"] = "sad"
        elif any(word in user_input_lower for word in ["happy", "joyful", "excited", "great"]):
            state["mood"] = "happy"
        elif any(word in user_input_lower for word in ["angry", "mad", "frustrated", "annoyed"]):
            state["mood"] = "angry"
        
        return state

    def _is_pure_memory_query(self, user_input: str) -> bool:
        """Check if this is a pure memory recall query (not mixed with other topics)."""
        user_input_lower = user_input.lower()
        
        # Comprehensive memory indicators
        memory_indicators = [
            'what is my', 'what are my', 'what do i', 'do you remember',
            'my favorite', 'what did i tell you', 'what do you know about me',
            'what is my favorite', 'what are my favorite', 'my favorite color',
            'my favorite food', 'what do i like', 'what do i love', 'what do i enjoy',
            'my name', 'my age', 'my job', 'my hobbies', 'my preferences',
            'what did i say', 'what did i mention', 'my choices', 'my decisions',
            'my habits', 'my style', 'my taste', 'my preferences'
        ]
        
        # Educational content indicators
        educational_indicators = [
            'how does', 'how do', 'explain', 'tell me about',
            'define', 'describe', 'why is', 'why are', 'when is', 'when are',
            'where is', 'where are', 'who is', 'who are'
        ]
        
        # Check for educational indicators
        has_educational_indicators = any(indicator in user_input_lower for indicator in educational_indicators)
        
        # Enhanced special case handling for "what is" and "what are"
        if 'what is' in user_input_lower or 'what are' in user_input_lower:
            # Comprehensive personal preference words
            personal_preference_words = [
                'my', 'favorite', 'color', 'food', 'like', 'love', 'enjoy', 'prefer',
                'name', 'age', 'job', 'hobbies', 'preferences', 'choices', 'decisions',
                'habits', 'style', 'taste', 'want', 'need', 'wish', 'hope'
            ]
            # If it contains personal preference words, it's NOT educational
            if not any(word in user_input_lower for word in personal_preference_words):
                has_educational_indicators = True
        
        # Check for memory indicators
        has_memory_indicators = any(indicator in user_input_lower for indicator in memory_indicators)
        
        # Pure memory query ONLY if it has memory indicators AND no educational indicators
        return has_memory_indicators and not has_educational_indicators

    def _recall_personal_facts(self, user_input: str) -> str:
        """Disabled - no personal data retrieval in ChatGPT-like mode."""
        return ""
    def _recall_most_recent_identity(self, user_input: str) -> str:
        """Disabled - no personal data retrieval in ChatGPT-like mode."""
        return ""

    def _get_conversation_context(self, user_input: str) -> str:
        """Get recent conversation context if relevant."""
        try:
            from memory import memory
            
            # ATOMIC MEMORY SNAPSHOT - Get consistent conversation state using public API
            knowledge_snapshot = memory.export('knowledge')
            conversation_memory = knowledge_snapshot.get('conversation_memory', [])
            
            if not conversation_memory:
                return ""
            
            recent_exchanges = []
            
            # Simplified conversation memory handling
            if isinstance(conversation_memory, list):
                recent_exchanges = conversation_memory[-3:]  # Last 3 exchanges
            elif isinstance(conversation_memory, dict):
                # Handle dict format - extract content if it's a list
                content = conversation_memory.get('content', [])
                if isinstance(content, list):
                    recent_exchanges = content[-3:]
                else:
                    recent_exchanges = []
            else:
                logger.warning(f"[Reasoner] Unexpected conversation memory format: {type(conversation_memory)}")
                return ""
            
            # Validate and clean up exchanges
            valid_exchanges = []
            for exchange in recent_exchanges:
                if isinstance(exchange, dict):
                    user_q = exchange.get('user', '')
                    ai_a = exchange.get('ai', '')
                    if user_q and ai_a:
                        valid_exchanges.append({
                            'user': str(user_q).strip(),
                            'ai': str(ai_a).strip()
                        })
            
            # Build context from valid exchanges
            context_parts = []
            for exchange in valid_exchanges[-3:]:  # Limit to last 3 valid exchanges
                user_q = exchange['user']
                ai_a = exchange['ai']
                
                # Truncate long responses
                if len(ai_a) > 100:
                    ai_a = ai_a[:100] + "..."
                if len(user_q) > 50:
                    user_q = user_q[:50] + "..."
                    
                context_parts.append(f"Q: {user_q} A: {ai_a}")
            
            return " | ".join(context_parts) if context_parts else ""
            
        except Exception as e:
            logger.error(f"[Reasoner] Error getting conversation context: {e}")
            return ""

    def _generate_llm_response_fast(self, prompt: str, memory_context: str = None) -> str:
        """Optimized LLM response with timeout and retry improvements."""
        # Quick response cache for common greetings and questions
        prompt_lower = prompt.lower().strip()
        quick_responses = {
            "hello": "Hello! How can I help you today?",
            "hi": "Hi there! What would you like to know?",
            "hey": "Hey! How are you doing?",
            "how are you": "I'm doing great, thanks for asking! How can I assist you?",
            "good morning": "Good morning! I hope you're having a wonderful day.",
            "good afternoon": "Good afternoon! How can I help you?",
            "good evening": "Good evening! What can I do for you?",
            "thanks": "You're very welcome!",
            "thank you": "You're very welcome!",
            "bye": "Goodbye! Have a great day!",
            "goodbye": "Goodbye! Take care!",
            "hey mira": "Hello! I'm Mira, your AI assistant. How can I help you today?",
            "hey mida": "Hello! I'm Mira, your AI assistant. How can I help you today?",
            "what is": "I'd be happy to explain that! Could you be more specific about what you'd like to know?",
            "explain": "I'd be happy to explain! What specifically would you like me to clarify?",
            "tell me about": "I'd love to tell you about that! What would you like to know?",
            "how does": "Great question! Let me explain how that works.",
            "why is": "That's an interesting question! Let me explain the reasoning behind that.",
            "what are": "I can help explain that! What specifically are you curious about?",
            "liquid hydrogen": "Liquid hydrogen is hydrogen gas that's been cooled to extremely low temperatures, around -253°C. It's used as rocket fuel and in various industrial applications because it's very energy-dense.",
            "hydrogen": "Hydrogen is the simplest and most abundant element in the universe. It's a colorless, odorless gas that's highly flammable and used in many industrial processes."
        }
        
        # Check for exact matches first
        if prompt_lower in quick_responses:
            logger.info("[Reasoner] Using cached quick response")
            return quick_responses[prompt_lower]
        
        # Check for partial matches (more restrictive to avoid overriding specific questions)
        for key, response in quick_responses.items():
            # Only use cached responses for very short, simple prompts
            if key in prompt_lower and len(prompt_lower) < 15 and not any(tech_word in prompt_lower for tech_word in ["bomb", "gamma", "ray", "physics", "quantum", "nuclear", "matter", "antimatter", "explain", "why", "how", "what", "differentiate"]):
                logger.info(f"[Reasoner] Using cached response for '{key}'")
                return response
        
        # Special case for hydrogen-related questions (only for simple queries)
        if any(word in prompt_lower for word in ["hydrogen", "liquid hydrogen", "h2"]) and len(prompt_lower) < 20:
            if "liquid" in prompt_lower:
                logger.info("[Reasoner] Using cached response for liquid hydrogen")
                return quick_responses["liquid hydrogen"]
            else:
                logger.info("[Reasoner] Using cached response for hydrogen")
                return quick_responses["hydrogen"]
        
        max_retries = 2  # Reduced from 3
        retry_delay = 0.5  # Reduced delay
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                self._performance_stats['llm_calls'] += 1
                
                # ChatGPT-like system prompt - simple and direct
                if memory_context:
                    system_prompt = (
                        "You are Mira, a helpful AI assistant. "
                        "Answer questions directly and clearly. "
                        "Be conversational and helpful.\n\n"
                        f"[CONTEXT]\n{memory_context}\n\n"
                        "Use the conversation context naturally when relevant."
                    )
                else:
                    system_prompt = (
                        "You are Mira, a helpful AI assistant. "
                        "Answer questions directly and clearly. "
                        "Be conversational and helpful."
                    )

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "keep_alive": 120,  # Further reduced for faster response
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more direct, factual responses
                        "top_k": 1,  # Very low for most direct responses
                        "top_p": 0.2,  # Lower for more focused responses
                        "num_predict": 120,  # Increased for longer responses
                        "num_ctx": 2048,  # Restore context for better understanding
                        "num_batch": 512,  # Restore batch size
                        "num_thread": 4,  # Restore threads
                        "repeat_penalty": 1.1,  # Prevent repetition
                        "stop": ["User:", "Human:", "Assistant:"]  # Removed \n\n to prevent early stopping
                    }
                }

                # Optimized timeout for speed
                resp = _ollama_session.post(OLLAMA_URL, json=payload, headers=OLLAMA_HEADERS, timeout=12)
                
                if resp.status_code != 200:
                    error_message = resp.json().get("error", "Unknown error")
                    raise requests.exceptions.HTTPError(f"Ollama error: {error_message}")
                
                data = resp.json()
                full_response = data.get("message", {}).get("content", "").strip()

                processing_time = time.time() - start_time
                if processing_time > 10.0:
                    logger.warning(f"[Reasoner] Slow LLM response: {processing_time:.2f}s")
                
                if full_response:
                    return full_response
                else:
                    raise ValueError("Empty response from Ollama")

            except requests.exceptions.Timeout:
                logger.warning(f"[Reasoner] LLM timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return "I'm taking longer than usual to respond. Please try a simpler question."
                    
            except Exception as e:
                logger.error(f"[Reasoner] LLM error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return "I'm having trouble generating a response right now. Please try again."
        
        return "I'm unable to respond at the moment. Please try again later."
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics for monitoring."""
        stats = self._performance_stats.copy()
        if stats['memory_operations'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['memory_operations']
        else:
            stats['avg_processing_time'] = 0.0
        
        cache_hit_rate = 0.0
        total_cache_ops = stats['cache_hits'] + stats['cache_misses']
        if total_cache_ops > 0:
            cache_hit_rate = stats['cache_hits'] / total_cache_ops
        stats['cache_hit_rate'] = cache_hit_rate
        
        # Add memory statistics
        try:
            from memory import memory
            memory_stats = memory.get_memory_stats()
            stats['memory_stats'] = memory_stats
        except Exception as e:
            stats['memory_stats'] = {'error': str(e)}
        
        return stats
    
    def clear_performance_stats(self):
        """Clear performance statistics."""
        self._performance_stats = {
            'memory_operations': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0
        }
        
        # Clear caches
        self._semantic_cache.clear()
        self._classification_cache.clear()
            
    def summarize_reasoning(self, user_input: str) -> str:
        """Print the complete reasoning chain for debugging."""
        logger.info("\n=== REASONING CHAIN ===")
        
        # Memory recall
        recalled_fact = recall_from_knowledge(user_input)
        logger.info(f"Memory Recall: {recalled_fact or 'No relevant memory'}")
        
        # Logic inference
        user_state = self._extract_user_state(user_input)
        context = {
            "user": {
                "input": user_input,
                "facts": recalled_fact or "",
                "skip_study": user_state.get("skip_study", False),
                "tired": user_state.get("tired", False),
                "mood": user_state.get("mood", "neutral")
            },
            "ai_response": "",
            "timestamp": time.time()
        }
        
        active_rules = rule_engine.get_active_rules(context)
        logger.info(f"Active Rules: {[rule.id for rule in active_rules]}")
        
        context = infer(context)
        triggered_rules = context.get('rules_triggered', [])
        logger.info(f"Triggered Rules: {triggered_rules}")
        
        # Emotional analysis
        emotion = self.emotion_reflector.infer_emotion(user_input)
        logger.info(f"Inferred Emotion: {emotion}")
        
        # Response source
        if context.get("ai_response"):
            logger.info("Response Source: Rule-based")
        else:
            logger.info("Response Source: LLM fallback")
        
        logger.info("=== END REASONING CHAIN ===\n")

if __name__ == "__main__":
    logger.info("=== Mira AI Reasoner Self-Test ===")
    reasoner = Reasoner(model="llama3")

    test_inputs = [
        "I think I will skip studying today.",
        "I'm really tired right now."
    ]

    for user_input in test_inputs:
        logger.info(f"\nUser: {user_input}")
        reasoner.summarize_reasoning(user_input)
        response = reasoner.process_input(user_input)
        logger.info(f"Mira: {response}")

    logger.info("\n=== All Tests Completed ===")