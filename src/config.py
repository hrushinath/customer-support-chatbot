"""
Configuration settings for the Customer Support Chatbot
Modify these settings to customize behavior
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
# Which embedding model to use
# Options:
#   - "all-MiniLM-L6-v2" (384 dims, fast, good for RAG) ✓ RECOMMENDED
#   - "all-mpnet-base-v2" (768 dims, slower, better quality)
#   - "BGE-base-en-v1.5" (768 dims, excellent for retrieval)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Device for embedding model
# Options: "cpu" (works on all laptops), "cuda" (NVIDIA GPU), "mps" (Apple Silicon)
EMBEDDING_DEVICE = "cpu"

# Batch size for embedding generation (higher = faster but more RAM)
EMBEDDING_BATCH_SIZE = 32

# ============================================================================
# VECTOR DATABASE CONFIGURATION
# ============================================================================
# Which vector DB to use
# Options: "faiss" (fast, standalone), "chroma" (feature-rich)
VECTOR_DB_TYPE = "faiss"

# Path to FAISS index
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "chatbot_faiss.index"

# Path to store chunk metadata
METADATA_PATH = VECTOR_STORE_DIR / "chunks_metadata.json"

# ============================================================================
# DOCUMENT CHUNKING CONFIGURATION
# ============================================================================
# How to split documents into chunks
# Optimal: 300-1000 tokens per chunk for RAG

# Chunk size in characters (roughly 4 chars = 1 token)
CHUNK_SIZE = 1000  # ~250 tokens

# Overlap between chunks (prevents cutting important info)
CHUNK_OVERLAP = 200  # 20% overlap

# Separator for splitting (try: "\n\n", "\n", ".")
CHUNK_SEPARATOR = "\n\n"

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
# Number of relevant chunks to retrieve per query
TOP_K_CHUNKS = 5

# Minimum similarity score (0-1) to consider a chunk relevant
RELEVANCE_THRESHOLD = 0.2  # Lowered to allow more matches

# Filter out very short chunks (likely unhelpful)
MIN_CHUNK_LENGTH = 50

# ============================================================================
# LLM CONFIGURATION (Ollama)
# ============================================================================
# Local LLM to use (must be pulled with Ollama first)
# Popular options:
#   - "mistral" (7B, fast, good quality) ✓ RECOMMENDED
#   - "neural-chat" (7B, optimized for chat)
#   - "llama2" (7B, general purpose)
#   - "dolphin-mixtral" (8x7B, very smart but slower)
#   - "phi" (2.7B, ultra-lightweight, CPU-friendly)
#   - "neural-chat:7b-v3.1-q4_0" (quantized, faster)

LLM_MODEL = "mistral"

# Ollama API endpoint
OLLAMA_BASE_URL = "http://localhost:11434"

# Ollama generation parameters
LLM_TEMPERATURE = 0.3  # Lower = more deterministic, Higher = more creative
LLM_TOP_P = 0.9  # Nucleus sampling (0-1)
LLM_MAX_TOKENS = 500  # Max length of response

# Timeout for LLM requests (seconds)
LLM_TIMEOUT = 120

# ============================================================================
# RAG PROMPT TEMPLATE
# ============================================================================
# System prompt that guides the LLM's behavior
RAG_SYSTEM_PROMPT = """You are a helpful and professional customer support assistant.
Your role is to answer customer questions accurately and courteously.

IMPORTANT RULES:
1. Answer ONLY using the provided context below
2. If the answer is not in the context, say: "I don't have information about that"
3. Be concise and clear in your responses
4. If you're unsure, express appropriate uncertainty
5. Always be respectful and professional

Do NOT:
- Make up information
- Provide generic responses not supported by context
- Pretend to know things outside the provided context"""

# User query template before context
QUERY_TEMPLATE = """Context information:
{context}

User Question: {question}

Please answer the question based ONLY on the context provided above."""

# ============================================================================
# RESPONSE FORMATTING
# ============================================================================
# Include source references in response?
INCLUDE_SOURCES = True

# How to determine confidence
# Options: "similarity_score", "llm_feedback", "both"
CONFIDENCE_METHOD = "similarity_score"

# ============================================================================
# DOCUMENT LOADING CONFIGURATION
# ============================================================================
# File extensions to process
SUPPORTED_EXTENSIONS = {
    ".txt": "text",
    ".pdf": "pdf",
    ".json": "json",
    ".docx": "docx",
    ".md": "text"
}

# Maximum file size to process (MB)
MAX_FILE_SIZE_MB = 50

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = LOGS_DIR / "chatbot.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================
# Use GPU acceleration if available (faster but requires CUDA)
USE_GPU = False

# Cache embeddings to disk for faster restarts
CACHE_EMBEDDINGS = True

# Number of workers for parallel document processing
NUM_WORKERS = 4

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================
# Re-ranking: Use a second model to re-rank top-K results
# Makes results better but slower (recommended for production)
USE_RERANKING = False

# Reranker model (requires transformers library)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Store chunk embeddings in memory (faster but uses more RAM)
# Disable if running on low-memory systems
CACHE_EMBEDDINGS_IN_MEMORY = True

# ============================================================================
# DEFAULT KNOWLEDGE BASE STRUCTURE
# ============================================================================
# Where to look for knowledge base files
KB_FAQS_DIR = KNOWLEDGE_BASE_DIR / "faqs"
KB_DOCUMENTS_DIR = KNOWLEDGE_BASE_DIR / "documents"

# ============================================================================
# HELPFUL PRINT STATEMENT
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CHATBOT CONFIGURATION")
    print("=" * 70)
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Vector DB Type: {VECTOR_DB_TYPE}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Ollama URL: {OLLAMA_BASE_URL}")
    print(f"Knowledge Base: {KNOWLEDGE_BASE_DIR}")
    print(f"Vector Store: {VECTOR_STORE_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
    print(f"Top-K Retrieval: {TOP_K_CHUNKS} chunks")
    print(f"GPU Enabled: {USE_GPU}")
    print("=" * 70)
