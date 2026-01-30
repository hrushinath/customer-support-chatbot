"""
Streamlit Web Interface for Customer Support Chatbot
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import CustomerSupportChatbot
import config

# Page configuration
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        color: #000000;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        color: #000000;
    }
    .bot-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
        color: #000000;
    }
    .message-content {
        color: #1a1a1a;
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "stats" not in st.session_state:
    st.session_state.stats = {
        "total_queries": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0
    }

def initialize_chatbot():
    """Initialize the chatbot"""
    if st.session_state.chatbot is None:
        with st.spinner("🔄 Initializing chatbot... This may take a moment..."):
            try:
                chatbot = CustomerSupportChatbot()
                if chatbot.initialize():
                    st.session_state.chatbot = chatbot
                    st.session_state.initialized = True
                    return True
                else:
                    st.error("❌ Failed to initialize chatbot. Check if Ollama is running.")
                    return False
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                return False
    return True

def reload_knowledge_base():
    """Reload the knowledge base"""
    if st.session_state.chatbot:
        with st.spinner("🔄 Reloading knowledge base..."):
            try:
                st.session_state.chatbot.reload_knowledge_base()
                st.success("✅ Knowledge base reloaded successfully!")
                st.session_state.messages = []  # Clear chat history
            except Exception as e:
                st.error(f"❌ Error reloading: {str(e)}")

def get_confidence_class(confidence):
    """Get CSS class for confidence level"""
    if confidence == "high":
        return "confidence-high"
    elif confidence == "medium":
        return "confidence-medium"
    else:
        return "confidence-low"

def display_message(role, content, confidence=None, sources=None):
    """Display a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div style='color: #000000;'><strong>👤 You:</strong></div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_html = ""
        if confidence:
            conf_class = get_confidence_class(confidence)
            confidence_html = f'<span class="{conf_class}">Confidence: {confidence.upper()}</span>'
        
        sources_html = ""
        if sources:
            sources_html = f"<div style='font-size:0.8rem; margin-top:0.5rem; color:#555;'>📚 Sources: {sources}</div>"
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div style='color: #000000;'><strong>🤖 Bot:</strong> {confidence_html}</div>
            <div class="message-content">{content}</div>
            {sources_html}
        </div>
        """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize Chatbot", type="primary"):
            initialize_chatbot()
    else:
        st.success("✅ Chatbot Ready")
    
    st.markdown("---")
    
    # Reload button
    if st.session_state.initialized:
        if st.button("🔄 Reload Knowledge Base"):
            reload_knowledge_base()
    
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    if st.session_state.initialized and st.session_state.chatbot:
        stats = st.session_state.chatbot.get_stats()
        st.metric("Vector DB Chunks", stats.get("vector_db_size", 0))
        st.metric("Total Queries", st.session_state.stats["total_queries"])
        
        if st.session_state.stats["total_queries"] > 0:
            st.markdown("**Confidence Distribution:**")
            st.write(f"🟢 High: {st.session_state.stats['high_confidence']}")
            st.write(f"🟡 Medium: {st.session_state.stats['medium_confidence']}")
            st.write(f"🔴 Low: {st.session_state.stats['low_confidence']}")
    
    st.markdown("---")
    
    # Configuration info
    st.markdown("### 🔧 Configuration")
    st.write(f"**LLM Model:** {config.LLM_MODEL}")
    st.write(f"**Embedding Model:** {config.EMBEDDING_MODEL.split('/')[-1]}")
    st.write(f"**Top-K Chunks:** {config.TOP_K_CHUNKS}")
    st.write(f"**Relevance Threshold:** {config.RELEVANCE_THRESHOLD}")
    
    st.markdown("---")
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    st.markdown("""
    - What is your return policy?
    - How long does shipping take?
    - Do you ship internationally?
    - What warranty do you provide?
    - How can I contact support?
    - What payment methods do you accept?
    """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main content
st.markdown('<div class="main-header">🤖 Customer Support Chatbot</div>', unsafe_allow_html=True)

if not st.session_state.initialized:
    st.info("👈 Click **Initialize Chatbot** in the sidebar to get started!")
    st.markdown("""
    ### Welcome to the Customer Support Chatbot!
    
    This AI-powered chatbot uses **Retrieval-Augmented Generation (RAG)** to answer your questions
    based on the knowledge base. 
    
    **Features:**
    - 🧠 Local LLM powered by Ollama (Mistral)
    - 🔍 Semantic search with FAISS vector database
    - 📚 Context-aware responses from knowledge base
    - 🔒 Fully offline and private
    
    **Before you start:**
    1. Make sure Ollama is running: `ollama serve`
    2. Ensure Mistral model is downloaded: `ollama pull mistral`
    3. Click **Initialize Chatbot** in the sidebar
    """)
else:
    # Display chat history
    for msg in st.session_state.messages:
        display_message(
            msg["role"],
            msg["content"],
            msg.get("confidence"),
            msg.get("sources")
        )
    
    # Chat input
    user_input = st.chat_input("Ask me anything about our products and services...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get bot response
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.chatbot.ask(user_input)
                
                # Update statistics
                st.session_state.stats["total_queries"] += 1
                confidence = result.get("confidence", "low")
                st.session_state.stats[f"{confidence}_confidence"] += 1
                
                # Format sources
                sources_text = None
                if result.get("sources"):
                    sources_list = [s.get("source", "unknown") for s in result["sources"][:3]]
                    sources_text = ", ".join([Path(s).name for s in sources_list])
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "confidence": confidence,
                    "sources": sources_text
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Powered by <strong>Mistral 7B</strong> + <strong>FAISS</strong> + <strong>Sentence Transformers</strong> 
    | Built with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
