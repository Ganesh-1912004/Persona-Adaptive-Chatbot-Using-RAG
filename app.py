# 📂 persona_adaptive_chatbot/
# └── 📄 app.py

import streamlit as st
import time
import os
from dotenv import load_dotenv

# --- Import Backend Modules ---
# Make sure the backend directory is in the Python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.behavioral_analyzer import analyze_behavior
from backend.persona_engine import get_or_create_persona, update_persona, save_persona, get_persona_summary
from backend.rag_handler import setup_rag_chain, get_rag_response

# --- Page Configuration ---
st.set_page_config(
    page_title="Persona-Adaptive AI",
    page_icon="🤖",
    layout="wide"
)

# --- Load Environment Variables ---
# Create a .env file in your project root and add your OpenAI API key
# OPENAI_API_KEY="sk-..."
load_dotenv()

# --- Functions ---

@st.cache_resource
def load_rag_chain():
    """Load and cache the RAG chain to avoid reloading on every interaction."""
    # Path to your knowledge base file
    knowledge_base_path = os.path.join("data", "knowledge_base.txt")
    if not os.path.exists(knowledge_base_path):
        st.error(f"Knowledge base file not found at {knowledge_base_path}")
        st.stop()
    return setup_rag_chain(knowledge_base_path)

def initialize_session_state():
    """Initialize variables in Streamlit's session state."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"

    if "persona" not in st.session_state:
        st.session_state.persona = get_or_create_persona(st.session_state.user_id)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]

    if "last_message_time" not in st.session_state:
        st.session_state.last_message_time = time.time()

    if "rag_chain" not in st.session_state:
        with st.spinner("Initializing AI model..."):
            st.session_state.rag_chain = load_rag_chain()

# --- Main Application ---

st.title("🤖 Persona-Adaptive Conversational AI")
st.caption("An AI that adapts to your personality and behavior in real-time.")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    st.warning("OpenAI API key is not set. Please add it to your .env file.", icon="⚠️")
    st.stop()

# Initialize session state
initialize_session_state()

# --- Sidebar for Persona Display ---
with st.sidebar:
    st.header("🧠 Dynamic User Persona")
    st.caption("This profile updates in real-time based on your interactions.")
    persona_summary_placeholder = st.empty()
    persona_summary_placeholder.markdown(get_persona_summary(st.session_state.persona))

# --- Chat Interface ---
# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Get user input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- Behavioral Analysis ---
    current_time = time.time()
    time_since_last_message = current_time - st.session_state.last_message_time
    st.session_state.last_message_time = current_time

    behavioral_data = analyze_behavior(prompt, time_since_last_message)

    # --- Persona Update ---
    st.session_state.persona = update_persona(st.session_state.persona, behavioral_data)
    save_persona(st.session_state.persona) # Save the updated persona
    persona_summary_placeholder.markdown(get_persona_summary(st.session_state.persona)) # Update sidebar

    # --- Get AI Response ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare context for the RAG chain
            chat_history = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages]
            
            # Get response from the RAG chain
            response = get_rag_response(
                rag_chain=st.session_state.rag_chain,
                question=prompt,
                chat_history=chat_history,
                persona=st.session_state.persona
            )
            
            st.write(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
