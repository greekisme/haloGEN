# Mira AI Persistent Memory System
# Structured, persistent, namespaced key-value store with semantic knowledge recall.
# This is the sole memory system used by Mira AI.

import os
import re
import hashlib
import time
import threading
import shutil
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import Counter
import math
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Memory configuration
MEMORY_LIMIT = 10  # Keep last 10 conversation exchanges
MEMORY_VERSION = 2  # Schema version for future migrations

# Global variables for semantic similarity
_similarity_threshold = 0.2  # Lower threshold for better recall while preventing irrelevant matches
_tfidf_cache = {}
_document_frequencies = {}
_normalized_cache = {}  # Cache for normalized TF-IDF vectors

# Synonym groups for better semantic matching
_synonym_groups = [
    {'pet', 'dog', 'cat', 'animal', 'puppy', 'kitten'},
    {'food', 'eat', 'pizza', 'chocolate', 'ice cream', 'meal', 'dinner', 'lunch'},
    {'color', 'blue', 'red', 'green', 'yellow', 'purple', 'orange'},
    {'like', 'love', 'enjoy', 'prefer', 'favorite'},
    {'read', 'reading', 'books', 'book'},
    {'season', 'summer', 'winter', 'spring', 'autumn', 'fall'}
]

class Memory:
    """Structured, persistent, namespaced key-value store with TTL support."""
    
    def __init__(self, data_dir: str = "./data", data_file: str = "memory.json"):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file)
        self._lock = threading.Lock()
        self._data = {}
        self._load()
    
    def _ensure_data_dir(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            logger.debug(f"Created data directory: {self.data_dir}")
    
    def _load(self):
        """Load data from persistent storage with atomic read."""
        self._ensure_data_dir()
        
        if not os.path.exists(self.data_file):
            self._data = {}
            self._save()
            return
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            logger.info(f"Loaded memory from {self.data_file}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Corrupt memory file, creating backup: {e}")
            self._backup_corrupt_file()
            self._data = {}
            self._save()
    
    def _backup_corrupt_file(self):
        """Backup corrupt file with timestamp."""
        if os.path.exists(self.data_file):
            timestamp = int(time.time())
            backup_path = os.path.join(self.data_dir, f"memory.corrupt.{timestamp}.json")
            shutil.move(self.data_file, backup_path)
            logger.warning(f"Backed up corrupt file to: {backup_path}")
    
    def _save(self):
        """Save data to persistent storage with atomic write."""
        self._ensure_data_dir()
        
        # Atomic write: write to temp file, then replace
        temp_file = self.data_file + ".tmp"
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.data_file)
            logger.debug(f"Saved memory to {self.data_file}")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _prune_expired(self, namespace: str = "global"):
        """Remove expired keys from a namespace.
        
        MUST be called with lock held - this method does not acquire locks internally.
        """
        if namespace not in self._data:
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._data[namespace].items():
            if isinstance(entry, dict) and 'expires_at' in entry:
                if current_time >= entry['expires_at']:
                    expired_keys.append(key)
        
        for key in expired_keys:
            del self._data[namespace][key]
            logger.debug(f"Pruned expired key: {namespace}:{key}")
        
        # Log cleanup statistics
        if expired_keys:
            logger.info(f"[Memory] Cleaned up {len(expired_keys)} expired entries from {namespace}")
    
    def cleanup_expired_entries(self, namespace: str = None):
        """Proactively clean up expired entries across all or specific namespace."""
        with self._lock:
            if namespace:
                self._prune_expired(namespace)
            else:
                # Clean all namespaces
                for ns in list(self._data.keys()):
                    self._prune_expired(ns)
    
    def cleanup_old_conversations(self, max_conversations: int = 10):
        """Clean up old conversation entries, keeping only the most recent ones."""
        with self._lock:
            conversation_memory = self._data.get('knowledge', {}).get('conversation_memory', [])
            
            if isinstance(conversation_memory, list) and len(conversation_memory) > max_conversations:
                # Keep only the most recent conversations
                old_count = len(conversation_memory)
                self._data['knowledge']['conversation_memory'] = conversation_memory[-max_conversations:]
                logger.info(f"[Memory] Cleaned up {old_count - max_conversations} old conversation entries")
    
    def cleanup_low_confidence_entries(self, min_confidence: float = 0.3):
        """Clean up entries with very low confidence scores."""
        with self._lock:
            total_cleaned = 0
            
            for namespace in list(self._data.keys()):
                if namespace == 'conversation_memory':
                    continue  # Skip conversation memory
                    
                expired_keys = []
                for key, entry in self._data[namespace].items():
                    if isinstance(entry, dict):
                        confidence = entry.get('confidence', 0.5)
                        if confidence < min_confidence:
                            expired_keys.append(key)
                
                for key in expired_keys:
                    del self._data[namespace][key]
                    total_cleaned += 1
                
                if expired_keys:
                    logger.info(f"[Memory] Cleaned up {len(expired_keys)} low-confidence entries from {namespace}")
            
            if total_cleaned > 0:
                logger.info(f"[Memory] Total low-confidence entries cleaned: {total_cleaned}")
    
    def cleanup_old_entries(self, max_age_hours: int = 24):
        """Remove entries older than specified hours."""
        with self._lock:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            removed_count = 0
            
            for namespace in list(self._data.keys()):
                if namespace in self._data and isinstance(self._data[namespace], dict):
                    expired_keys = []
                    for key, entry in self._data[namespace].items():
                        if isinstance(entry, dict):
                            entry_time = entry.get('timestamp', current_time)
                            if current_time - entry_time > max_age_seconds:
                                expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self._data[namespace][key]
                        removed_count += 1
                    
                    if expired_keys:
                        logger.info(f"[Memory] Cleaned up {len(expired_keys)} old entries from {namespace}")
            
            if removed_count > 0:
                logger.info(f"[Memory] Total old entries cleaned: {removed_count}")
                self._save()
    
    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage and cleanup opportunities."""
        with self._lock:
            stats = {
                'total_namespaces': len(self._data),
                'total_entries': 0,
                'expired_entries': 0,
                'low_confidence_entries': 0,
                'conversation_entries': 0,
                'by_type': {}
            }
            
            current_time = time.time()
            
            for namespace, entries in self._data.items():
                if isinstance(entries, dict):
                    stats['total_entries'] += len(entries)
                    
                    for key, entry in entries.items():
                        if isinstance(entry, dict):
                            # Count by type
                            entry_type = entry.get('type', 'unknown')
                            stats['by_type'][entry_type] = stats['by_type'].get(entry_type, 0) + 1
                            
                            # Count expired entries
                            if 'expires_at' in entry and current_time >= entry['expires_at']:
                                stats['expired_entries'] += 1
                            
                            # Count low confidence entries
                            confidence = entry.get('confidence', 0.5)
                            if confidence < 0.3:
                                stats['low_confidence_entries'] += 1
                            
                            # Count conversation entries
                            if entry_type == 'conversation':
                                stats['conversation_entries'] += 1
                
                elif isinstance(entries, list) and namespace == 'conversation_memory':
                    stats['conversation_entries'] = len(entries)
            
            return stats
    
    def get(self, key: str, default=None, namespace: str = "global"):
        """Get a value from the store."""
        with self._lock:
            self._prune_expired(namespace)
            
            if namespace not in self._data:
                return default
            
            if key not in self._data[namespace]:
                return default
            
            entry = self._data[namespace][key]
            
            # Handle backward compatibility
            # For structured entries, return the content for backward compatibility
            # External systems can use export() method to get full metadata
            if isinstance(entry, dict) and 'content' in entry:
                return entry['content']
            else:
                # For simple values, return as-is
                return entry
    
    def set(self, key: str, value: Any, namespace: str = "global", ttl_seconds: Optional[int] = None,
            item_type: str = "general", confidence: float = 0.5):
        """Set a value in the store with optional type and confidence metadata."""
        with self._lock:
            if namespace not in self._data:
                self._data[namespace] = {}
            
            # Create structured entry with metadata
            if isinstance(value, dict) and 'content' in value:
                # Already structured, use as-is
                entry = value
            else:
                # Create structured entry
                entry = {
                    'content': value,
                    'type': item_type,
                    'confidence': confidence,
                    'timestamp': time.time()
                }
            
            if ttl_seconds is not None:
                entry['expires_at'] = time.time() + ttl_seconds
            
            self._data[namespace][key] = entry
            self._save()
            logger.info(f"[Memory] Stored entry: type={entry.get('type')} conf={entry.get('confidence')}")
            logger.debug(f"Set {namespace}:{key} = {value} (type: {item_type}, confidence: {confidence})")
    
    def delete(self, key: str, namespace: str = "global") -> bool:
        """Delete a key from the store."""
        with self._lock:
            if namespace not in self._data:
                return False
            
            if key not in self._data[namespace]:
                return False
            
            del self._data[namespace][key]
            self._save()
            logger.debug(f"Deleted {namespace}:{key}")
            return True
    
    def exists(self, key: str, namespace: str = "global") -> bool:
        """Check if a key exists."""
        with self._lock:
            self._prune_expired(namespace)
            
            if namespace not in self._data:
                return False
            
            return key in self._data[namespace]
    
    def keys(self, namespace: str = "global") -> List[str]:
        """Get all keys in a namespace."""
        with self._lock:
            self._prune_expired(namespace)
            
            if namespace not in self._data:
                return []
            
            return list(self._data[namespace].keys())
    
    def export(self, namespace: Optional[str] = None) -> Dict:
        """Export data (entire store or a namespace)."""
        with self._lock:
            if namespace is None:
                return dict(self._data)
            return dict(self._data.get(namespace, {}))
    
    def import_data(self, data: Dict, merge: bool = True):
        """Import data into the store."""
        with self._lock:
            if merge:
                for namespace, keys in data.items():
                    if namespace not in self._data:
                        self._data[namespace] = {}
                    self._data[namespace].update(keys)
            else:
                self._data.update(data)
            
            self._save()
            logger.debug(f"Imported data, merge={merge}")

# Create singleton instance
memory = Memory()

# Public API functions
def load_knowledge() -> Dict:
    """Load knowledge from structured memory."""
    facts = memory.get('facts', {}, 'knowledge')
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    return {
        'facts': facts,
        'conversation_memory': conversation_memory
    }

def save_knowledge(data: Dict) -> bool:
    """Save knowledge to structured memory."""
    memory.set('facts', data.get('facts', {}), 'knowledge')
    memory.set('conversation_memory', data.get('conversation_memory', []), 'knowledge')
    return True

def add_to_knowledge(user_input: str) -> Optional[str]:
    """Detect phrases like 'remember that...' and store facts."""
    remember_patterns = [
        r"remember\s+that\s+(.+)",
        r"remember\s+(.+)",
        r"don't\s+forget\s+that\s+(.+)",
        r"don't\s+forget\s+(.+)",
        r"note\s+that\s+(.+)",
        r"note\s+(.+)"
    ]
    
    user_input_lower = user_input.lower().strip()
    
    for pattern in remember_patterns:
        match = re.search(pattern, user_input_lower, re.IGNORECASE)
        if match:
            fact = match.group(1).strip()
            if fact and _is_valid_fact(fact):
                # Store with timestamp-based key
                fact_key = f"fact_{int(time.time() * 1000)}"
                memory.set(fact_key, {
                    "content": fact,
                    "original_input": user_input,
                    "timestamp": time.time()
                }, 'knowledge_facts')
                logger.info(f"Added fact: {fact}")
                return fact
    return None

def recall_from_knowledge(query: str) -> Optional[str]:
    """Robust semantic recall with comprehensive error handling and data validation."""
    try:
        # Validate input
        if not isinstance(query, str) or not query.strip():
            logger.warning(f"[Memory] Invalid query: {type(query)}")
            return None
        
        # Safely gather all stored facts with comprehensive validation
        facts = {}
        
        # Process knowledge namespace using public API
        knowledge_data = memory.export('knowledge')
        if isinstance(knowledge_data, dict):
            for key, item_data in knowledge_data.items():
                try:
                    # Skip non-dict entries
                    if not isinstance(item_data, dict):
                        logger.warning(f"[Memory] Skipping non-dict entry: {key} -> {type(item_data)}")
                        continue
                    
                    # Skip entries without content
                    if 'content' not in item_data:
                        continue
                    
                    content = item_data['content']
                    
                    # Handle different content types
                    if isinstance(content, list):
                        # Flatten list content to string
                        content = ' '.join(str(item) for item in content if item)
                        logger.debug(f"[Memory] Flattened list content for {key}")
                    elif not isinstance(content, str):
                        # Convert other types to string
                        content = str(content)
                        logger.debug(f"[Memory] Converted {type(item_data['content'])} to string for {key}")
                    
                    # Skip empty content
                    if not content.strip():
                        continue
                    
                    # Store validated entry
                    facts[key] = {
                        'content': content,
                        'type': item_data.get('type', 'general'),
                        'confidence': item_data.get('confidence', 0.5),
                        'timestamp': item_data.get('timestamp', 0),
                        'correction': item_data.get('correction', '')
                    }
                    
                except Exception as item_err:
                    logger.warning(f"[Memory] Skipped malformed entry {key}: {item_err}")
                    continue
        
        # Process knowledge_facts namespace using public API
        fact_data_ns = memory.export('knowledge_facts')
        if isinstance(fact_data_ns, dict):
            for key, item_data in fact_data_ns.items():
                try:
                    if not isinstance(item_data, dict) or 'content' not in item_data:
                        continue
                    
                    content = item_data['content']
                    if isinstance(content, list):
                        content = ' '.join(str(item) for item in content if item)
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    if not content.strip():
                        continue
                    
                    facts[key] = {
                        'content': content,
                        'type': item_data.get('type', 'general'),
                        'confidence': item_data.get('confidence', 0.5),
                        'timestamp': item_data.get('timestamp', 0),
                        'correction': item_data.get('correction', '')
                    }
                    
                except Exception as item_err:
                    logger.warning(f"[Memory] Skipped malformed fact {key}: {item_err}")
                    continue
        
        if not facts:
            return None
        
        query_lower = query.lower()
        
        # Check for revisit requests
        revisit_phrases = [
            'revisit last topic', 'continue previous', 'what was my last question',
            'previous topic', 'last conversation', 'what did we talk about',
            'continue from before', 'pick up where we left off', 'my last question',
            'previous conversation', 'what did i ask', 'what did we discuss',
            'last thing i asked', 'what was the last question', 'last question i asked',
            'what did i just ask', 'my previous question', 'what was my previous question',
            'last thing we talked about', 'what did we just discuss', 'previous discussion'
        ]
        if any(phrase in query_lower for phrase in revisit_phrases):
            return revisit_last_conversation()
        
        # Direct keyword matching for specific queries (higher priority)
        direct_matches = []
        for key, fact in facts.items():
            content_lower = fact['content'].lower()
            
            # Check for exact keyword matches
            if 'favorite color' in query_lower and 'favorite color' in content_lower:
                direct_matches.append((key, fact, 1.0))  # Highest priority
            elif 'name' in query_lower and ('name' in content_lower or 'my name' in content_lower):
                direct_matches.append((key, fact, 0.9))
            elif 'pizza' in query_lower and 'pizza' in content_lower:
                direct_matches.append((key, fact, 0.9))
            elif 'love' in query_lower and 'love' in content_lower:
                direct_matches.append((key, fact, 0.8))
        
        # If we have direct matches, return the best one
        if direct_matches:
            direct_matches.sort(key=lambda x: x[2], reverse=True)  # Sort by priority
            best_match = direct_matches[0][1]
            logger.info(f"[Memory] Direct match found for query: {query}")
            return best_match['content']
        
        # Compute semantic similarity with error handling
        try:
            _update_document_frequencies(facts)
            query_tfidf = _compute_tfidf(query)
        except Exception as tfidf_err:
            logger.error(f"[Memory] TF-IDF computation failed: {tfidf_err}")
            return None
        
        matches = []
        for fact_key, fact_data in facts.items():
            try:
                content = fact_data['content']
                
                # Skip very long content that might cause issues
                if len(content) > 1000:
                    logger.debug(f"[Memory] Skipping very long content: {fact_key}")
                    continue
                
                fact_tfidf = _compute_tfidf(content)
                similarity = _cosine_similarity_tfidf(query_tfidf, fact_tfidf)
                
                if similarity >= _similarity_threshold:
                    matches.append({
                        "content": content,
                        "similarity": similarity,
                        "confidence": fact_data.get("confidence", 0.5),
                        "timestamp": fact_data.get("timestamp", 0),
                        "type": fact_data.get("type", "general"),
                        "correction": fact_data.get("correction", "")
                    })
                    
            except Exception as match_err:
                logger.warning(f"[Memory] Skipped fact {fact_key} during matching: {match_err}")
                continue
        
        if not matches:
            return None
        
        # Sort by type priority, then confidence, then timestamp
        type_priority = {"user_identity": 3, "personal_preference": 2, "factual_claim": 1, "belief": 0, "general": 0}
        matches.sort(key=lambda x: (type_priority.get(x.get("type", "general"), 0), x.get("confidence", 0), x.get("timestamp", 0)), reverse=True)
        best_match = matches[0]
        content = best_match["content"]
        
        # Clean up content - remove various prefixes
        content_lower = content.lower()
        if content_lower.startswith("remember that "):
            content = content[14:]
        elif content_lower.startswith("remember "):
            content = content[9:]
        elif content_lower.startswith("and remember that "):
            content = content[17:]
        elif content_lower.startswith("and remember "):
            content = content[13:]
        
        natural_fact = _convert_to_second_person(content)
        
        # Generate response based on type
        if best_match["type"] == "belief" and best_match.get("correction"):
            return f"You once said you believe {natural_fact}. {best_match['correction']}"
        elif best_match["type"] == "belief":
            return f"You once said you believe {natural_fact}."
        elif best_match["type"] == "user_identity":
            # Extract just the name part
            if 'name is' in natural_fact.lower():
                name_part = natural_fact.lower().split('name is')[-1].strip().rstrip('.')
                return f"Your name is {name_part}."
            elif 'live in' in natural_fact.lower():
                location_part = natural_fact.lower().split('live in')[-1].strip().rstrip('.')
                return f"You live in {location_part}."
            else:
                return f"Your name is {natural_fact}."
        elif best_match["type"] == "personal_preference":
            # Extract just the preference part
            if 'favorite color is' in natural_fact.lower():
                color_part = natural_fact.lower().split('favorite color is')[-1].strip().rstrip('.')
                return f"Your favorite color is {color_part}."
            else:
                return f"Your favorite color is {natural_fact}."
        else:
            similarity = best_match.get("similarity", 0)
            if similarity >= 0.6:
                return f"As I recall, you once told me that {natural_fact}."
            elif similarity >= 0.5:
                return f"You mentioned earlier that {natural_fact}."
            else:
                return f"As I remember, {natural_fact}."
    
    except Exception as e:
        logger.error(f"[Memory] Critical error in semantic recall: {e}")
        # Fallback to keyword recall if TF-IDF fails using public API
        knowledge_data = memory.export('knowledge')
        return _fallback_keyword_recall(query, knowledge_data)


def update_memory(user_input: str, ai_output: str) -> None:
    """Store recent dialogue context (limit last 10 messages)."""
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    
    # Handle both dict and list formats
    if isinstance(conversation_memory, dict):
        # If it's a dict, extract the content
        content = conversation_memory.get('content', '')
        if isinstance(content, str):
            # Try to parse as JSON if it's a string
            try:
                import json
                conversation_memory = json.loads(content)
            except:
                conversation_memory = []
        else:
            conversation_memory = content if isinstance(content, list) else []
    elif not isinstance(conversation_memory, list):
        conversation_memory = []
    
    # Ensure it's a list before appending
    if isinstance(conversation_memory, list):
        conversation_memory.append({"user": user_input, "ai": ai_output})
        
        # Keep only the last N exchanges
        if len(conversation_memory) > MEMORY_LIMIT:
            conversation_memory = conversation_memory[-MEMORY_LIMIT:]
        
        # Store back as structured entry
        memory.set('conversation_memory', {
            "content": conversation_memory,
            "type": "conversation",
            "confidence": 1.0,
            "timestamp": time.time()
        }, "knowledge")
        
        logger.debug(f"Updated conversation memory: {len(conversation_memory)} exchanges")

def build_prompt(user_input: str) -> str:
    """Build a prompt string combining system prompt, memory context, and new input."""
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    conversation_context = ""
    if conversation_memory:
        conversation_context = "\n\nRecent conversation:\n"
        for entry in conversation_memory:
            conversation_context += f"User: {entry['user']}\n"
            conversation_context += f"Mira: {entry['ai']}\n"
    
    # Combine system prompt with conversation context
    full_prompt = f"{os.environ.get('SYSTEM_PROMPT', 'You are Mira, a helpful AI assistant.')}{conversation_context}"
    return full_prompt

def revisit_last_conversation() -> Optional[str]:
    """Return the last question-answer pair from conversation_memory."""
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    
    if not conversation_memory:
        return "I don't have any previous conversation to revisit yet."
    
    # Get the last exchange
    last_exchange = conversation_memory[-1]
    user_question = last_exchange.get('user', '')
    ai_answer = last_exchange.get('ai', '')
    
    if user_question and ai_answer:
        # Truncate long responses for better readability
        if len(ai_answer) > 150:
            ai_answer = ai_answer[:150] + "..."
        
        return f"You asked: '{user_question}' and I responded: '{ai_answer}'. Would you like me to continue from there?"
    
    return "I don't have a complete previous conversation to revisit."

def _is_personal_preference_fact(fact_content: str) -> bool:
    """Detect if a fact is about personal preferences."""
    personal_indicators = [
        'favorite', 'like', 'love', 'enjoy', 'prefer', 'hate', 'dislike',
        'my favorite', 'i like', 'i love', 'i enjoy', 'i prefer',
        'you like', 'you love', 'you enjoy', 'you prefer'
    ]
    fact_lower = fact_content.lower()
    return any(indicator in fact_lower for indicator in personal_indicators)

def _is_factual_question(query: str) -> bool:
    """Detect if a query is asking for factual information."""
    factual_indicators = [
        'what is', 'what are', 'how does', 'how do', 'why is', 'why are',
        'when is', 'when are', 'where is', 'where are', 'who is', 'who are',
        'explain', 'define', 'describe', 'tell me about', 'what does',
        'how does', 'what causes', 'what makes', 'what happens'
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in factual_indicators)

def _is_personal_question(query: str) -> bool:
    """Detect if a query is asking about personal preferences."""
    personal_indicators = [
        'my favorite', 'what do i like', 'what do i love', 'my preference',
        'what do i enjoy', 'what do i prefer', 'my favorite color',
        'my favorite food', 'what do i remember', 'do you remember',
        'what is my favorite', 'what are my favorite', 'my favorite',
        'do you know my', 'what do you know about me', 'tell me about me'
    ]
    query_lower = query.lower()
    return any(indicator in query_lower for indicator in personal_indicators)

def clear_memory() -> None:
    """Clear all conversation memory."""
    memory.set('conversation_memory', [], 'knowledge')

def get_memory_stats() -> Dict:
    """Get memory statistics."""
    facts = memory.get('facts', {}, 'knowledge')
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    return {
        'total_facts': len(facts),
        'conversation_exchanges': len(conversation_memory),
        'memory_limit': MEMORY_LIMIT
    }

# Helper functions for debugging
def get_fact_count() -> int:
    """Get total number of stored facts."""
    facts = memory.get('facts', {}, 'knowledge')
    knowledge_facts = memory.keys('knowledge_facts')
    return len(facts) + len(knowledge_facts)

def get_recent_conversations(limit: int = 5) -> List[Dict]:
    """Get recent conversation exchanges."""
    conversation_memory = memory.get('conversation_memory', [], 'knowledge')
    return conversation_memory[-limit:] if conversation_memory else []

def summarize_memory() -> str:
    """Return a summary of memory contents."""
    stats = get_memory_stats()
    fact_count = get_fact_count()
    recent_conversations = get_recent_conversations(3)
    
    summary = f"Memory Summary:\n"
    summary += f"- Total facts: {fact_count}\n"
    summary += f"- Conversation exchanges: {stats['conversation_exchanges']}\n"
    
    if recent_conversations:
        summary += f"- Recent conversations: {len(recent_conversations)} exchanges\n"
    
    # Check for emotional reflection
    emotional_reflection = memory.get('emotional_reflection', namespace='knowledge')
    if emotional_reflection:
        summary += f"- Last emotional reflection: {emotional_reflection[:100]}...\n"
    
    return summary

# Helper functions for semantic similarity
def _expand_synonyms(words: List[str]) -> List[str]:
    """Expand words with their synonyms for better semantic matching."""
    expanded_words = list(words)
    
    for word in words:
        for synonym_group in _synonym_groups:
            if word in synonym_group:
                # Add all synonyms from the group
                expanded_words.extend(synonym_group)
                break
    
    return expanded_words

def _preprocess_text(text: str) -> List[str]:
    """Preprocess text for semantic analysis."""
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out very short words and common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours',
        'hers', 'ours', 'theirs', 'am', 'are', 'is', 'was', 'were', 'be', 'been', 'being'
    }
    filtered_words = [word for word in words if len(word) >= 3 and word not in stop_words]
    
    # Expand with synonyms
    return _expand_synonyms(filtered_words)

def _compute_tfidf(text: str) -> Dict[str, float]:
    """Compute TF-IDF vector for text with caching optimization."""
    global _tfidf_cache, _document_frequencies, _normalized_cache
    
    if text in _tfidf_cache:
        return _tfidf_cache[text]
    
    words = _preprocess_text(text)
    if not words:
        return {}
    
    # Compute term frequencies
    word_counts = Counter(words)
    total_words = len(words)
    tf = {word: count / total_words for word, count in word_counts.items()}
    
    # Compute IDF (simplified version)
    idf = {}
    for word in word_counts:
        if word in _document_frequencies:
            idf[word] = math.log(len(_document_frequencies) / _document_frequencies[word])
        else:
            idf[word] = 1.0
    
    # Compute TF-IDF
    tfidf = {word: tf[word] * idf[word] for word in tf}
    
    # Cache normalized vector for efficiency
    _tfidf_cache[text] = tfidf
    return tfidf

def _cosine_similarity_tfidf(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    if not vec1 or not vec2:
        return 0.0
    
    # Get all unique words
    all_words = set(vec1.keys()) | set(vec2.keys())
    
    # Compute dot product and magnitudes
    dot_product = sum(vec1.get(word, 0) * vec2.get(word, 0) for word in all_words)
    magnitude1 = math.sqrt(sum(vec1.get(word, 0) ** 2 for word in all_words))
    magnitude2 = math.sqrt(sum(vec2.get(word, 0) ** 2 for word in all_words))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)

def _update_document_frequencies(facts: Dict):
    """Safe IDF calculation that skips non-string or malformed entries."""
    global _document_frequencies
    _document_frequencies = {}

    for fact_key, fact_data in facts.items():
        try:
            content = fact_data.get('content', '')
            if not isinstance(content, str) or not content.strip():
                continue
            words = _preprocess_text(content)
            for word in set(words):
                _document_frequencies[word] = _document_frequencies.get(word, 0) + 1
        except Exception as e:
            logger.warning(f"[Memory] Skipped invalid fact {fact_key}: {e}")


def _is_valid_fact(text: str) -> bool:
    """
    Returns True if text is declarative (not a question/command).
    """
    t = text.lower().strip()
    if t.endswith("?") or t.startswith(("who", "what", "when", "where", "why", "how")):
        return False
    if "?" in t or "!" in t:
        return False
    # Accept declarative statements with "is/are" or common preference patterns
    return (" is " in t or " are " in t or 
            any(word in t for word in ["love", "like", "enjoy", "prefer", "hate", "dislike"]))

def _fact_relevance_score(text: str, stored_fact: str) -> float:
    """Compute word overlap ratio for fuzzy relevance."""
    t1 = set(text.lower().split())
    t2 = set(stored_fact.lower().split())
    return len(t1 & t2) / max(1, len(t1 | t2))

def _fallback_keyword_recall(query: str, facts: Dict) -> Optional[str]:
    """Fallback keyword-based recall if semantic similarity fails."""
    query_lower = query.lower()
    query_keywords = _extract_keywords(query_lower)
    if not query_keywords:
        return None
    
    best_match = None
    best_score = 0
    best_fact_content = ""
    
    for fact_key, fact_data in facts.items():
        fact_content = fact_data.get('content', '').lower()
        fact_keywords = _extract_keywords(fact_content)
        
        # Calculate relevance score based on keyword overlap
        overlap = len(query_keywords.intersection(fact_keywords))
        
        # Also check for partial matches and related concepts
        partial_score = 0
        for q_word in query_keywords:
            for f_word in fact_keywords:
                # Check for exact word matches (highest priority)
                if q_word == f_word:
                    partial_score += 1.0
                # Check for partial word matches
                elif q_word in f_word or f_word in q_word:
                    partial_score += 0.3
                # Check for semantic relationships
                elif _are_related_words(q_word, f_word):
                    partial_score += 0.5
        
        total_score = overlap + partial_score
        if total_score > best_score and total_score > 0:
            best_score = total_score
            best_match = fact_content
            best_fact_content = fact_data.get('content', '')
    
    # Only return a match if there's significant keyword overlap
    if best_match and best_score >= 1.0:
        natural_fact = _convert_to_second_person(best_fact_content)
        return f"As I remember, {natural_fact}."
    
    return None

def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords (nouns, important words) from text."""
    # Enhanced stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours',
        'hers', 'ours', 'theirs', 'am', 'are', 'is', 'was', 'were', 'be', 'been', 'being',
        'what', 'who', 'when', 'where', 'why', 'how', 'which', 'tell', 'about', 'me',
        'just', 'like', 'get', 'got', 'go', 'goes', 'going', 'went', 'come', 'came',
        'see', 'saw', 'know', 'knows', 'think', 'thinks', 'want', 'wants', 'need', 'needs',
        'make', 'makes', 'take', 'takes', 'give', 'gives', 'say', 'says', 'said', 'tell',
        'tells', 'told', 'ask', 'asks', 'asked', 'help', 'helps', 'work', 'works',
        'time', 'times', 'day', 'days', 'way', 'ways', 'thing', 'things', 'people',
        'person', 'man', 'woman', 'child', 'children', 'here', 'there', 'now', 'then',
        'very', 'really', 'quite', 'so', 'too', 'also', 'well', 'good', 'bad', 'great',
        'nice', 'fine', 'okay', 'ok', 'yes', 'no', 'maybe', 'sure', 'right', 'wrong'
    }
    
    # Extract words and filter
    words = set(text.split())
    keywords = words - stop_words
    
    # Remove punctuation from keywords
    import string
    keywords = {word.strip(string.punctuation) for word in keywords}
    
    # Filter out very short words (less than 3 characters) and common filler words
    keywords = {word for word in keywords if len(word) >= 3 and word not in {
        'um', 'uh', 'ah', 'oh', 'hmm', 'well', 'so', 'like', 'you', 'know'
    }}
    
    return keywords

def _are_related_words(word1: str, word2: str) -> bool:
    """Check if two words are semantically related (basic synonyms and related concepts)."""
    # Basic synonym and related concept mappings
    related_groups = [
        {'eat', 'food', 'pizza', 'meal', 'dinner', 'lunch', 'breakfast', 'snack'},
        {'color', 'colour', 'blue', 'red', 'green', 'yellow', 'purple', 'orange'},
        {'dog', 'pet', 'animal', 'puppy', 'canine'},
        {'sun', 'star', 'solar', 'sunlight', 'bright'},
        {'like', 'love', 'enjoy', 'prefer', 'favorite', 'favourite'},
        {'name', 'named', 'called', 'nickname'},
        {'short', 'brief', 'quick', 'fast', 'concise'},
        {'answer', 'response', 'reply', 'solution'}
    ]
    
    word1_lower = word1.lower()
    word2_lower = word2.lower()
    
    # Check if both words are in the same related group
    for group in related_groups:
        if word1_lower in group and word2_lower in group:
            return True
    
    return False

def _convert_to_second_person(text: str) -> str:
    """Convert first person statements to second person for natural conversation."""
    import re
    
    # Use regex with word boundaries for more precise matching
    result = text
    
    # Apply conversions with word boundaries
    result = re.sub(r'\bi am\b', 'you are', result)
    result = re.sub(r"\bi'm\b", "you're", result)
    result = re.sub(r'\bmine\b', 'yours', result)
    result = re.sub(r'\bmy\b', 'your', result)
    result = re.sub(r'\bme\b', 'you', result)
    result = re.sub(r'\bi\b', 'you', result)
    
    return result

# Self-test section
if __name__ == "__main__":
    print("=== Mira AI Memory System Self-Test ===")
    
    # Test 1: TTL functionality
    print("1. Testing TTL functionality...")
    memory.set("test_key", "test_value", "test_namespace", ttl_seconds=1)
    assert memory.get("test_key", namespace="test_namespace") == "test_value"
    print("   ✓ TTL set successfully")
    
    time.sleep(1.1)
    assert memory.get("test_key", namespace="test_namespace") is None
    print("   ✓ TTL expiry working")
    
    # Test 2: Persistence
    print("2. Testing persistence...")
    memory.set("persistent_key", "persistent_value", "test_namespace")
    memory.set("another_key", {"nested": "data"}, "test_namespace")
    
    # Create new instance to test persistence
    test_memory = Memory()
    assert test_memory.get("persistent_key", namespace="test_namespace") == "persistent_value"
    assert test_memory.get("another_key", namespace="test_namespace") == {"nested": "data"}
    print("   ✓ Persistence working")
    
    # Test 3: Export/Import
    print("3. Testing export/import...")
    export_data = memory.export("test_namespace")
    assert "persistent_key" in export_data
    assert "another_key" in export_data
    print("   ✓ Export working")
    
    # Test 4: Namespace operations
    print("4. Testing namespace operations...")
    memory.set("key1", "value1", "namespace1")
    memory.set("key2", "value2", "namespace2")
    
    keys1 = memory.keys("namespace1")
    keys2 = memory.keys("namespace2")
    
    assert "key1" in keys1
    assert "key2" in keys2
    assert memory.exists("key1", "namespace1")
    assert not memory.exists("key1", "namespace2")
    print("   ✓ Namespace operations working")
    
    # Test 5: Knowledge functions
    print("5. Testing knowledge functions...")
    add_result = add_to_knowledge("Remember that I love pizza")
    assert add_result == "i love pizza"
    print("   ✓ Add knowledge working")
    
    recall_result = recall_from_knowledge("What do I like to eat?")
    assert recall_result is not None
    print("   ✓ Recall knowledge working")
    
    # Test 6: Helper functions
    print("6. Testing helper functions...")
    fact_count = get_fact_count()
    recent_conversations = get_recent_conversations()
    summary = summarize_memory()
    print(f"   ✓ Fact count: {fact_count}")
    print(f"   ✓ Recent conversations: {len(recent_conversations)}")
    print(f"   ✓ Memory summary generated")
    
    # Cleanup
    memory.delete("persistent_key", "test_namespace")
    memory.delete("another_key", "test_namespace")
    memory.delete("key1", "namespace1")
    memory.delete("key2", "namespace2")
    
    print("\n=== All Tests Passed! ===")
    print(f"Using structured persistent memory at: {memory.data_file}")
    print("Memory system is ready for Mira AI!")