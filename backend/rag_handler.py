# 📂 persona_adaptive_chatbot/
# └── 📁 backend/
#     └── 📄 rag_handler.py

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def setup_vector_store(file_path: str):
    """Loads a text file, splits it, creates embeddings, and sets up a FAISS vector store."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore.as_retriever()

def create_persona_aware_prompt(persona: dict) -> PromptTemplate:
    """Creates a dynamic prompt template based on the user's persona."""
    
    # Determine the tone based on persona
    sentiment = persona['emotional_state']['current_sentiment']
    style = persona['communication_style']

    tone_instruction = "Your tone should be helpful and neutral."
    if sentiment == 'positive':
        tone_instruction = "Your tone should be upbeat and encouraging."
    elif sentiment == 'negative':
        tone_instruction = "Your tone should be calm, empathetic, and reassuring."

    # Determine the response length
    length_instruction = "Provide a comprehensive answer."
    if style == 'brief':
        length_instruction = "Keep your answer concise and to the point."
    elif style == 'detailed':
        length_instruction = "Provide a detailed, in-depth answer, explaining the concepts clearly."

    # Get top topics to prime the context
    top_topics = sorted(persona["contextual_preferences"]["topic_interests"].items(), key=lambda item: item[1], reverse=True)
    topic_hint = ""
    if top_topics:
        topic_hint = f"The user is often interested in topics like {', '.join([t[0] for t in top_topics[:3]])}."

    template_string = f"""
You are a persona-adaptive AI assistant. Your goal is to answer the user's question accurately based on the provided context, while adapting your communication style to the user's current persona.

**User Persona Analysis:**
- **Current Emotion:** {sentiment.capitalize()}
- **Preferred Style:** {style.capitalize()}
- {topic_hint}

**Your Instructions:**
- {tone_instruction}
- {length_instruction}
- Answer the question based ONLY on the following context.
- If the context doesn't contain the answer, say "I'm sorry, I don't have enough information to answer that."

**Context:**
{{context}}

**Chat History:**
{{chat_history}}

**User's Question:**
{{question}}

**Your Answer:**
"""
    return PromptTemplate(template=template_string, input_variables=["context", "question", "chat_history"])


def setup_rag_chain(file_path: str):
    """Initializes the entire RAG chain with a static retriever."""
    retriever = setup_vector_store(file_path)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    
    # The chain will be constructed dynamically in the response function
    # based on the latest persona.
    return {"retriever": retriever, "llm": llm}


def get_rag_response(rag_chain: dict, question: str, chat_history: list, persona: dict) -> str:
    """
    Gets a response from the RAG chain using the dynamic, persona-aware prompt.
    """
    retriever = rag_chain['retriever']
    llm = rag_chain['llm']
    
    # Create the persona-aware prompt for this specific interaction
    prompt_template = create_persona_aware_prompt(persona)
    
    # Construct the chain for this specific call
    rag_chain_dynamic = (
        {"context": retriever, "question": RunnablePassthrough(), "chat_history": RunnablePassthrough(), "persona": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Invoke the chain with all necessary inputs
    response = rag_chain_dynamic.invoke({
        "question": question,
        "chat_history": "\n".join(chat_history),
        "persona": persona # Pass the whole persona object if the prompt needs more details
    })
    
    return response
