# Usage Guide

## 🎯 Running the Chatbot

### Interactive Mode (Default)

```bash
python src/app.py
```

**Output:**
```
ℹ️  Initializing CustomerSupportChatbot...
ℹ️  Starting initialization...
ℹ️  Loading embedding model...
ℹ️  Vector DB: 47 chunks
ℹ️  ✓ Initialization complete!

======================================================================
CUSTOMER SUPPORT CHATBOT
======================================================================
Type 'quit' or 'exit' to close
Type 'stats' to see statistics
Type 'reload' to rebuild knowledge base
======================================================================

You: What is your return policy?

Bot: We accept returns within 30 days of purchase. The item must be in 
its original condition with all tags attached. You'll need a receipt or 
proof of purchase.

Confidence: high

Sources:
  - general_faq.json (chunk 0, similarity: 0.92)
```

### Interactive Commands

- **Ask questions**: Type any customer question
- **`stats`**: View chatbot statistics and configuration
- **`reload`**: Rebuild vector index from knowledge base
- **`quit` or `exit`**: Close the chatbot

---

## 🐍 Using in Python Code

### Basic Usage

```python
from src.app import CustomerSupportChatbot

# Initialize chatbot
chatbot = CustomerSupportChatbot(verbose=True)
if not chatbot.initialize():
    print("Failed to initialize")
    exit(1)

# Ask a question
response = chatbot.ask("What is your return policy?")

# Print response
print(response['answer'])
print(f"Confidence: {response['confidence']}")
```

### Advanced Usage

```python
# Get response with detailed information
response = chatbot.ask(
    question="How long does shipping take?",
    return_sources=True
)

# Response structure
print(f"Question: {response['question']}")
print(f"Answer: {response['answer']}")
print(f"Confidence: {response['confidence']}")  # high/medium/low
print(f"Sources: {response['sources']}")  # List of source documents
print(f"Context chunks: {response.get('context_chunks', 0)}")
print(f"Avg similarity: {response.get('avg_similarity', 0):.2f}")

# Access statistics
stats = chatbot.get_stats()
print(f"Vector DB size: {stats['vector_db_size']}")
print(f"Embedding model: {stats['embedding_model']}")
print(f"LLM model: {stats['llm_model']}")
```

### Batch Processing

```python
# Ask multiple questions
questions = [
    "What is your return policy?",
    "How long does shipping take?",
    "Do you ship internationally?",
    "What payment methods do you accept?"
]

responses = chatbot.batch_ask(questions)

# Process responses
for resp in responses:
    print(f"Q: {resp['question']}")
    print(f"A: {resp['answer']}\n")
```

---

## 🔌 Web API (FastAPI)

### Create API Server

Create `api_server.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.app import CustomerSupportChatbot

app = FastAPI(title="Customer Support Chatbot")
chatbot = None

@app.on_event("startup")
async def startup():
    global chatbot
    chatbot = CustomerSupportChatbot(verbose=False)
    if not chatbot.initialize():
        raise RuntimeError("Failed to initialize chatbot")

class QuestionRequest(BaseModel):
    question: str
    include_sources: bool = True

class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    sources: list = []

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if not chatbot or not chatbot.is_initialized:
        raise HTTPException(status_code=503, detail="Chatbot not ready")
    
    response = chatbot.ask(
        request.question,
        return_sources=request.include_sources
    )
    
    if not response.get('success', True):
        raise HTTPException(
            status_code=400,
            detail=response.get('error', 'Unknown error')
        )
    
    return response

@app.get("/stats")
async def get_stats():
    if not chatbot or not chatbot.is_initialized:
        raise HTTPException(status_code=503, detail="Chatbot not ready")
    return chatbot.get_stats()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "chatbot_ready": chatbot is not None and chatbot.is_initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Run Server

```bash
# Install FastAPI
pip install fastapi uvicorn

# Run server
python api_server.py
```

### Test API

```bash
# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is your return policy?"}'

# Get statistics
curl http://localhost:8000/stats

# Health check
curl http://localhost:8000/health
```

### Python Client

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "How long does shipping take?"}
)

data = response.json()
print(data['answer'])
```

---

## 🎨 Streamlit UI

### Create Web Interface

Create `streamlit_app.py`:

```python
import streamlit as st
from src.app import CustomerSupportChatbot

# Page config
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize chatbot (cached)
@st.cache_resource
def load_chatbot():
    chatbot = CustomerSupportChatbot(verbose=False)
    chatbot.initialize()
    return chatbot

# Title
st.title("🤖 Customer Support Chatbot")
st.write("Ask any question about our products and services!")

# Load chatbot
with st.spinner("Loading chatbot..."):
    chatbot = load_chatbot()

# Main chat interface
question = st.text_input("Your Question:", placeholder="e.g., What is your return policy?")

if question:
    with st.spinner("Thinking..."):
        response = chatbot.ask(question, return_sources=True)
    
    # Display answer
    st.success("✅ Answer")
    st.write(response['answer'])
    
    # Display confidence and sources
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Confidence", response.get('confidence', 'Unknown'))
    
    with col2:
        st.metric("Context Chunks", response.get('context_chunks', 0))
    
    # Display sources
    if response.get('sources'):
        with st.expander("📚 Sources"):
            for source in response['sources']:
                st.write(f"**{source['source']}** (chunk {source['chunk_id']})")
                st.write(f"Similarity: {source['similarity']:.4f}")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Statistics
    if st.button("Show Statistics"):
        stats = chatbot.get_stats()
        st.json(stats)
    
    # Reload knowledge base
    if st.button("Reload Knowledge Base"):
        with st.spinner("Rebuilding..."):
            chatbot.reload_knowledge_base()
        st.success("✅ Reloaded!")
    
    # About
    st.markdown("---")
    st.markdown("""
    ### About
    This chatbot uses **Retrieval-Augmented Generation (RAG)** to provide
    accurate, grounded answers based on your knowledge base.
    
    - No hallucinations
    - Source attribution
    - Completely local
    """)
```

### Run Streamlit App

```bash
# Install Streamlit
pip install streamlit

# Run app
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 🧪 Testing and Evaluation

### Create Test Suite

Create `test_chatbot.py`:

```python
import unittest
from src.app import CustomerSupportChatbot

class TestChatbot(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize chatbot once for all tests"""
        cls.chatbot = CustomerSupportChatbot(verbose=False)
        assert cls.chatbot.initialize(), "Failed to initialize"
    
    def test_initialization(self):
        """Test chatbot initializes correctly"""
        self.assertTrue(self.chatbot.is_initialized)
        self.assertIsNotNone(self.chatbot.embedding_generator)
        self.assertIsNotNone(self.chatbot.vector_store)
    
    def test_simple_question(self):
        """Test answering a simple question"""
        response = self.chatbot.ask("What is your return policy?")
        self.assertTrue(response['success'])
        self.assertIn('answer', response)
        self.assertGreater(len(response['answer']), 0)
    
    def test_confidence_scores(self):
        """Test confidence scoring"""
        response = self.chatbot.ask("What is shipping?")
        self.assertIn(response['confidence'], ['high', 'medium', 'low'])
    
    def test_source_attribution(self):
        """Test that sources are provided"""
        response = self.chatbot.ask("What is your warranty?", return_sources=True)
        self.assertTrue(len(response['sources']) > 0)
    
    def test_unknown_question(self):
        """Test handling of questions outside knowledge base"""
        response = self.chatbot.ask("What is quantum physics?")
        # Should either return low confidence or no context
        self.assertTrue(response['success'])

if __name__ == '__main__':
    unittest.main()
```

### Run Tests

```bash
python -m pytest test_chatbot.py -v
```

---

## 📊 Evaluation Metrics

### Manual Evaluation Checklist

For each question, check:

```
✓ Relevance: Does answer address the question?
✓ Accuracy: Is the information correct?
✓ Completeness: Does it cover all relevant aspects?
✓ Clarity: Is it well-written and understandable?
✓ Sources: Are sources relevant and accurate?
```

### Automated Metrics

```python
def evaluate_chatbot(test_questions, expected_answers):
    """Evaluate chatbot on test set"""
    scores = []
    
    for question, expected in zip(test_questions, expected_answers):
        response = chatbot.ask(question)
        
        # Check if answer contains key concepts from expected
        keywords = expected.split()
        matches = sum(1 for kw in keywords if kw.lower() in response['answer'].lower())
        relevance_score = matches / len(keywords)
        
        scores.append({
            'question': question,
            'score': relevance_score,
            'confidence': response['confidence']
        })
    
    avg_score = sum(s['score'] for s in scores) / len(scores)
    print(f"Average Score: {avg_score:.2%}")
    
    return scores
```

---

## 🛠️ Customization

### Custom Prompts

Edit `src/config.py`:

```python
RAG_SYSTEM_PROMPT = """You are a specialized customer support bot.
Be concise and professional. Always cite sources."""

QUERY_TEMPLATE = """Based on this information:
{context}

Answer this question: {question}"""
```

### Custom Chunking

In `src/app.py`:

```python
from src.modules.text_chunker import SmartChunker

# Use smart chunker (preserves sentence boundaries)
chunker = SmartChunker(chunk_size=800, chunk_overlap=150)
```

### Custom Embedding Model

In `src/config.py`:

```python
# Better quality (slower)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Lightweight (faster)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

---

## 📈 Monitoring

### Log Analysis

```python
import json
from pathlib import Path

# Read logs
log_file = Path("logs/chatbot.log")
with open(log_file) as f:
    for line in f:
        # Parse and analyze logs
        if "ERROR" in line:
            print(line)
```

### Performance Monitoring

```python
import time

start = time.time()
response = chatbot.ask("How long does shipping take?")
elapsed = time.time() - start

print(f"Response time: {elapsed:.2f}s")
print(f"Answer length: {len(response['answer'])} chars")
print(f"Context chunks: {response.get('context_chunks', 0)}")
```

---

**Next Steps:** See [OPTIMIZATION.md](OPTIMIZATION.md) for performance tuning!


<!-- FAQ's 
From FAQs (general_faq.json):

What is your return policy?
How long does shipping take?
What payment methods do you accept?
How do I track my order?
Do you offer international shipping?
What is your warranty policy?
How can I contact customer support?
Are your products eco-friendly? -->