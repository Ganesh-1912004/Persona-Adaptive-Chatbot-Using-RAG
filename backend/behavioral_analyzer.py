# 📂 persona_adaptive_chatbot/
# └── 📁 backend/
#     └── 📄 behavioral_analyzer.py

import time
from textblob import TextBlob

def analyze_typing_speed(time_diff_seconds: float) -> str:
    """
    Simulates typing speed analysis based on time between messages.
    In a real web app, this would be replaced with frontend JavaScript
    that captures actual keystroke timings.
    """
    if time_diff_seconds < 2:
        return "very_fast"
    elif time_diff_seconds < 5:
        return "fast"
    elif time_diff_seconds < 15:
        return "moderate"
    else:
        return "slow"

def detect_emotion(text: str) -> dict:
    """
    Detects emotion from text using TextBlob for sentiment analysis.
    Polarity: [-1, 1] (negative to positive)
    Subjectivity: [0, 1] (objective to subjective)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.3:
        sentiment = "positive"
    elif polarity < -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {"polarity": round(polarity, 2), "sentiment": sentiment}

def map_context(text: str) -> list:
    """
    Extracts key topics (nouns) from the text to understand the context.
    This is a simplified approach; more advanced methods like NER could be used.
    """
    blob = TextBlob(text)
    # Filter for nouns and exclude common words
    common_words = {"user", "chatbot", "question", "answer", "information"}
    topics = [word.lower() for word, pos in blob.tags if pos.startswith('NN') and word.lower() not in common_words]
    return list(set(topics))

def analyze_behavior(user_input: str, time_since_last_message: float) -> dict:
    """
    Main function to run all behavioral analyses and return a consolidated dictionary.
    """
    typing_speed = analyze_typing_speed(time_since_last_message)
    emotion = detect_emotion(user_input)
    topics = map_context(user_input)
    
    return {
        "typing_speed": typing_speed,
        "emotion": emotion,
        "topics": topics,
        "message_length": len(user_input.split())
    }
