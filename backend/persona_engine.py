# 📂 persona_adaptive_chatbot/
# └── 📁 backend/
#     └── 📄 persona_engine.py

import os
import json
from collections import deque

# --- Constants ---
PERSONA_DIR = "personas"
HISTORY_LENGTH = 10 # How many of the last data points to keep for averaging

# --- Persona Data Structure ---
def create_new_persona(user_id: str) -> dict:
    """Creates a default persona structure for a new user."""
    if not os.path.exists(PERSONA_DIR):
        os.makedirs(PERSONA_DIR)
    
    return {
        "user_id": user_id,
        "communication_style": "neutral", # Adapts to formal/informal
        "emotional_state": {
            "current_sentiment": "neutral",
            "sentiment_history": list(deque(maxlen=HISTORY_LENGTH))
        },
        "behavioral_patterns": {
            "avg_typing_speed": "moderate",
            "speed_history": list(deque(maxlen=HISTORY_LENGTH)),
            "avg_message_length": 20,
            "length_history": list(deque(maxlen=HISTORY_LENGTH))
        },
        "contextual_preferences": {
            "topic_interests": {}, # Stores topics and their frequency
        },
        "interaction_count": 0
    }

# --- Persona Management ---
def get_persona_filepath(user_id: str) -> str:
    """Constructs the file path for a user's persona data."""
    return os.path.join(PERSONA_DIR, f"{user_id}_persona.json")

def get_or_create_persona(user_id: str) -> dict:
    """Loads a persona from a file or creates a new one if it doesn't exist."""
    filepath = get_persona_filepath(user_id)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            persona_data = json.load(f)
            # Ensure deques are properly loaded if needed, though lists are fine for JSON
            return persona_data
    return create_new_persona(user_id)

def save_persona(persona: dict):
    """Saves the persona dictionary to its JSON file."""
    filepath = get_persona_filepath(persona["user_id"])
    with open(filepath, 'w') as f:
        json.dump(persona, f, indent=4)

# --- Persona Adaptation ---
def update_persona(persona: dict, behavioral_data: dict) -> dict:
    """Updates the persona based on new behavioral data."""
    # Update interaction count
    persona["interaction_count"] += 1

    # Update emotional state
    emotion = behavioral_data["emotion"]
    persona["emotional_state"]["current_sentiment"] = emotion["sentiment"]
    # Use a simple list to act like a deque for JSON serialization
    sentiment_history = persona["emotional_state"]["sentiment_history"]
    sentiment_history.append(emotion["polarity"])
    if len(sentiment_history) > HISTORY_LENGTH:
        sentiment_history.pop(0)
    persona["emotional_state"]["sentiment_history"] = sentiment_history

    # Update behavioral patterns
    speed_map = {"very_fast": 4, "fast": 3, "moderate": 2, "slow": 1}
    speed_history = persona["behavioral_patterns"]["speed_history"]
    speed_history.append(speed_map.get(behavioral_data["typing_speed"], 2))
    if len(speed_history) > HISTORY_LENGTH:
        speed_history.pop(0)
    persona["behavioral_patterns"]["speed_history"] = speed_history
    
    # Determine average speed
    avg_speed_val = sum(speed_history) / len(speed_history) if speed_history else 2
    if avg_speed_val > 3.5: persona["behavioral_patterns"]["avg_typing_speed"] = "very_fast"
    elif avg_speed_val > 2.5: persona["behavioral_patterns"]["avg_typing_speed"] = "fast"
    elif avg_speed_val > 1.5: persona["behavioral_patterns"]["avg_typing_speed"] = "moderate"
    else: persona["behavioral_patterns"]["avg_typing_speed"] = "slow"

    length_history = persona["behavioral_patterns"]["length_history"]
    length_history.append(behavioral_data["message_length"])
    if len(length_history) > HISTORY_LENGTH:
        length_history.pop(0)
    persona["behavioral_patterns"]["length_history"] = length_history
    persona["behavioral_patterns"]["avg_message_length"] = round(sum(length_history) / len(length_history), 1) if length_history else 20


    # Update communication style based on message length
    if persona["behavioral_patterns"]["avg_message_length"] > 30:
        persona["communication_style"] = "detailed"
    elif persona["behavioral_patterns"]["avg_message_length"] < 10:
        persona["communication_style"] = "brief"
    else:
        persona["communication_style"] = "neutral"

    # Update topic interests
    for topic in behavioral_data["topics"]:
        persona["contextual_preferences"]["topic_interests"][topic] = persona["contextual_preferences"]["topic_interests"].get(topic, 0) + 1

    return persona

def get_persona_summary(persona: dict) -> str:
    """Generates a markdown summary of the current persona state."""
    summary = f"""
    - **Emotion**: `{persona['emotional_state']['current_sentiment'].capitalize()}`
    - **Comm. Style**: `{persona['communication_style'].capitalize()}`
    - **Avg. Speed**: `{persona['behavioral_patterns']['avg_typing_speed'].replace('_', ' ').capitalize()}`
    - **Avg. Length**: `{persona['behavioral_patterns']['avg_message_length']}` words
    """
    
    # Get top 3 topics
    topics = persona["contextual_preferences"]["topic_interests"]
    if topics:
        top_topics = sorted(topics.items(), key=lambda item: item[1], reverse=True)[:3]
        summary += "\n- **Top Topics**:\n"
        for topic, _ in top_topics:
            summary += f"  - `{topic.capitalize()}`\n"
            
    return summary
