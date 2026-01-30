# Customer Support Chatbot with RAG - Architecture Guide

## 🎯 Overview

This document explains the architecture of a **Retrieval-Augmented Generation (RAG) based chatbot** that runs completely locally without any cloud services or paid APIs.

---

## 📚 What is RAG (Retrieval-Augmented Generation)?

### The Problem Without RAG
Without RAG, LLMs have limitations:
- **Hallucinations**: Generate plausible-sounding but false information
- **Outdated Knowledge**: Trained data becomes stale
- **No Private Data Access**: Can't use your specific knowledge base
- **No Source Attribution**: Can't tell where information came from

### How RAG Solves This

RAG works in 3 simple steps:

```
1. RETRIEVAL  → Find relevant documents from your knowledge base
2. CONTEXT    → Combine retrieved text with the user's question
3. GENERATION → Let the LLM answer using ONLY that context
```

**Result**: Grounded answers that cite sources and avoid hallucinations.

---

## 🏗️ System Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                       │
│                    (Web UI / CLI / API)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1. QUERY EMBEDDING                             │
│              (Convert text to vector)                            │
│      Model: Sentence-Transformers (all-MiniLM-L6)               │
│      Input: "What is your return policy?"                        │
│      Output: [0.23, -0.45, 0.67, ...] (384 dimensions)         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. VECTOR DATABASE SEARCH                        │
│             (Find similar documents in DB)                       │
│         Database: FAISS or ChromaDB (local files)                │
│         Similarity Metric: Cosine Similarity                     │
│         Returns: Top-K most relevant chunks (k=5)                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            3. CONTEXT CONSTRUCTION                               │
│        (Build the RAG prompt with context)                       │
│                                                                  │
│  Prompt Template:                                                │
│  ┌──────────────────────────────────────────┐                   │
│  │ You are a helpful customer support bot.  │                   │
│  │                                          │                   │
│  │ Use ONLY this context to answer:        │                   │
│  │ [Retrieved document chunks]              │                   │
│  │                                          │                   │
│  │ Question: [User query]                   │                   │
│  │ Answer:                                  │                   │
│  └──────────────────────────────────────────┘                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│            4. LLM RESPONSE GENERATION                            │
│              (Generate grounded answer)                          │
│    Models: Mistral / LLaMA 3 / Phi-3 (via Ollama)               │
│    Constraints: Only use provided context                        │
│    Output: Natural language response                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              5. RESPONSE FORMATTING                              │
│                                                                  │
│  {                                                               │
│    "question": "What is your return policy?",                   │
│    "answer": "We accept returns within 30 days...",             │
│    "confidence": "high",                                         │
│    "sources": [                                                  │
│      {"file": "policies.txt", "chunk_id": 3},                   │
│      {"file": "faq.json", "chunk_id": 1}                        │
│    ]                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. **Document Ingestion Module**
**File**: `src/modules/document_loader.py`

**What it does**:
- Loads documents from various sources (TXT, PDF, JSON, DOCX)
- Extracts text content
- Tracks document metadata

**Technologies**:
- PyPDF2 / pdfplumber (PDF files)
- python-docx (Word documents)
- json / csv (Structured data)

**Workflow**:
```
Raw Files → Extract Text → Store with Metadata
```

---

### 2. **Text Chunking Module**
**File**: `src/modules/text_chunker.py`

**What it does**:
- Splits long documents into chunks (500-1000 tokens)
- Maintains context by overlapping chunks
- Creates unique chunk IDs

**Why chunking matters**:
- LLMs have context limits
- Vector DBs work better with smaller, focused chunks
- Enables precise source attribution

**Example**:
```
Original: "Return Policy: Items can be returned within 30 days. 
Conditions: Must have receipt, tags attached, unworn/unused. 
Refunds processed in 5-7 business days."

Chunk 1: "Return Policy: Items can be returned within 30 days."
Chunk 2: "Conditions: Must have receipt, tags attached, unworn/unused."
Chunk 3: "Refunds processed in 5-7 business days."
```

---

### 3. **Embedding Module**
**File**: `src/modules/embeddings.py`

**What it does**:
- Converts text chunks into numerical vectors
- Uses sentence-level embeddings
- Stores embeddings with associated chunks

**Model**: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional vectors
- Fast (< 100ms per chunk)
- Excellent for semantic similarity

**How embeddings work**:
```
Text: "What is your refund policy?"
     ↓
[0.45, -0.23, 0.89, 0.12, ..., 0.34] (384 numbers)
     ↑
These numbers capture semantic meaning
"refund" and "return" will have similar vectors
```

---

### 4. **Vector Database Module**
**File**: `src/modules/vector_database.py`

**What it does**:
- Stores embeddings and chunks
- Enables fast similarity search
- Persists data locally

**Options**:
- **FAISS** (Facebook AI Similarity Search)
  - Ultra-fast, standalone
  - Perfect for laptops
  - No server needed
  
- **ChromaDB**
  - More feature-rich
  - Better for production
  - Easier metadata handling

**Storage**: Local `.faiss` files (no cloud upload)

---

### 5. **Query Processing Module**
**File**: `src/modules/query_processor.py`

**What it does**:
1. Embeds user query (same model as chunks)
2. Searches vector DB for top-K similar chunks
3. Ranks results by relevance
4. Assembles context for LLM

**Configuration**:
```python
k = 5  # Return top-5 chunks
threshold = 0.6  # Minimum relevance score
```

---

### 6. **Response Generation Module**
**File**: `src/modules/response_generator.py`

**What it does**:
- Constructs RAG prompt
- Calls local LLM (via Ollama)
- Formats response with confidence score
- Tracks sources

**LLM Options**:
- **Mistral 7B**: Fast, accurate, good for RAG
- **LLaMA 3 8B**: Better quality, more resources
- **Phi-3 3.8B**: Ultra-lightweight, CPU-friendly

---

## 🗂️ Project Structure

```
customer-support-chatbot/
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration settings
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── document_loader.py         # Load documents
│   │   ├── text_chunker.py            # Chunk documents
│   │   ├── embeddings.py              # Generate embeddings
│   │   ├── vector_database.py         # FAISS/ChromaDB wrapper
│   │   ├── query_processor.py         # Retrieve relevant chunks
│   │   └── response_generator.py      # Generate LLM responses
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                  # Logging setup
│   │   └── helpers.py                 # Utility functions
│   │
│   └── app.py                         # Main application
│
├── knowledge_base/
│   ├── faqs/
│   │   ├── general_faq.json
│   │   └── product_faq.json
│   │
│   └── documents/
│       ├── return_policy.txt
│       ├── shipping_info.pdf
│       └── product_manual.docx
│
├── vector_store/                      # Vector DB storage
│   ├── faiss_index.index
│   └── metadata.json
│
├── logs/                              # Application logs
│   └── chatbot.log
│
├── requirements.txt                   # Python dependencies
├── setup.sh / setup.bat               # Installation script
├── ARCHITECTURE.md                    # This file
├── SETUP_GUIDE.md                     # Installation guide
├── USAGE_GUIDE.md                     # How to use
└── README.md                          # Quick start
```

---

## 💾 Data Flow: Initialization Phase

When you first run the chatbot:

```
1. LOAD DOCUMENTS
   knowledge_base/ 
   → document_loader.py
   → Extract all text

2. CHUNK TEXT
   Raw text (10,000 words)
   → text_chunker.py
   → 15 chunks (500-800 words each)

3. GENERATE EMBEDDINGS
   Each chunk
   → Sentence-Transformers
   → 384-dimensional vector

4. STORE IN VECTOR DB
   Chunks + Embeddings
   → FAISS index
   → vector_store/faiss_index.index
```

**One-time operation** - happens during initialization, then reused.

---

## 💬 Data Flow: Query Phase

When user asks a question:

```
1. USER QUERY
   "What's your return policy?"

2. EMBED QUERY
   → Same embedding model
   → [0.45, -0.23, 0.89, ...] (384 dims)

3. SEARCH VECTOR DB
   Query vector
   → Cosine similarity search
   → Top-5 chunks with scores:
      ├─ Chunk #3 (score: 0.92)
      ├─ Chunk #7 (score: 0.88)
      ├─ Chunk #2 (score: 0.85)
      ├─ Chunk #9 (score: 0.79)
      └─ Chunk #1 (score: 0.76)

4. BUILD CONTEXT
   System Prompt:
   "You are a helpful customer support bot.
    Answer ONLY using the provided context.
    
    Context:
    [Retrieved chunks combined]
    
    Question: What's your return policy?
    
    Answer:"

5. CALL LOCAL LLM (Ollama)
   Prompt → Mistral 7B
   ↓
   "We accept returns within 30 days..."

6. RETURN RESPONSE
   {
     "question": "What's your return policy?",
     "answer": "We accept returns within...",
     "confidence": "high",
     "sources": ["policies.txt:chunk_3", "faq.json:chunk_7"]
   }
```

---

## 🛠️ Technology Choices Explained

### Why Sentence-Transformers?
- **Semantic Understanding**: Understands meaning, not just keywords
- **Lightweight**: 384 dimensions vs 4096+ for large models
- **Fast**: CPU inference < 100ms
- **Free**: Open-source, no API costs

### Why FAISS?
- **Speed**: Searches billions of vectors in milliseconds
- **Offline**: Works locally without internet
- **Scalable**: Handles 1M+ vectors on laptop
- **Minimal Dependencies**: Pure C++ under the hood

### Why Ollama?
- **Local LLM Running**: Simplest way to run models locally
- **GPU/CPU Flexible**: Works on any hardware
- **Model Manager**: Easy to switch between models
- **No Configuration**: Just run `ollama run mistral`

---

## 🎯 Key Concepts Summary

| Concept | Explanation | Benefit |
|---------|-------------|---------|
| **Embedding** | Text converted to numbers (vector) | Enables similarity search |
| **Vector DB** | Database of embeddings | Fast retrieval of similar docs |
| **RAG Prompt** | Template including context + question | Grounds LLM responses |
| **Chunking** | Breaking docs into pieces | Better retrieval accuracy |
| **Cosine Similarity** | Measure of vector closeness (0-1) | Relevance scoring |
| **Top-K Retrieval** | Return top 5 most similar chunks | Balance quality & speed |
| **Confidence Score** | How certain the answer is | User trust indicator |

---

## 🚀 Performance Characteristics

### Latency (per query)
- Embedding query: **10-50ms**
- Vector DB search: **5-20ms**
- LLM generation: **1-5 seconds** (depends on answer length)
- **Total: 1-5.5 seconds** per query

### Memory Usage
- Embedding model: **100-200 MB**
- Vector DB (1000 chunks): **50-100 MB**
- LLM model in memory: **3-16 GB** (depends on model size)
- **Total: 3.2-16.3 GB**

### Storage
- Models: **3-15 GB** (one-time download)
- Vector index: **100 MB** (per 10,000 chunks)
- Knowledge base: **50-500 MB** (depends on docs)

### Laptop Compatibility
✅ Works on: 
- **8GB RAM** (with Phi-3 3.8B)
- **16GB RAM** (with Mistral 7B or LLaMA 3 8B)
- **CPU-only laptops** (Intel i5+ / AMD Ryzen 5+)
- **With GPU**: 2-4x faster

---

## 🔐 Security & Privacy

### Data Stays Local
✅ No data sent to cloud
✅ No API calls to external services
✅ Complete privacy for sensitive docs
✅ GDPR/compliance friendly

### No Model Training
✅ Uses pre-trained models
✅ Your docs don't modify models
✅ Safe for confidential information

---

## Next Steps

1. **Setup**: Follow [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. **Run**: Follow [USAGE_GUIDE.md](USAGE_GUIDE.md)
3. **Customize**: Modify prompts in [src/config.py](src/config.py)
4. **Deploy**: See production tips in [OPTIMIZATION.md](OPTIMIZATION.md)

---

**Questions?** See detailed implementation in the code files or refer to the API documentation.
