# Optimization and Tuning Guide

## 🎯 Performance Optimization

### Scenario 1: Lightweight Laptop (8GB RAM, CPU only)

**Goal:** Balance quality and performance on limited hardware

```python
# src/config.py

# Use lightweight embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 16  # Smaller batches for low memory

# Use lightweight LLM
LLM_MODEL = "phi"  # Only 3.8B parameters

# Reduce context for faster processing
TOP_K_CHUNKS = 3
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Faster LLM settings
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 300
LLM_TIMEOUT = 60

# Disable memory-intensive features
CACHE_EMBEDDINGS_IN_MEMORY = False
USE_RERANKING = False
```

**Performance:**
- Model load: ~10 seconds
- Query response: 2-4 seconds
- Memory usage: 4-6 GB

### Scenario 2: Desktop (16GB+ RAM, with GPU)

**Goal:** Maximum quality with available resources

```python
# src/config.py

# Use high-quality embedding model
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DEVICE = "cuda"  # Use GPU
EMBEDDING_BATCH_SIZE = 64  # Large batches for GPU

# Use larger, better LLM
LLM_MODEL = "mistral"  # Good balance: 7B, fast, accurate
# Or: "neural-chat" for specialized chat

# More context for better answers
TOP_K_CHUNKS = 7
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

# High-quality settings
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 500
LLM_TIMEOUT = 120

# Enable advanced features
CACHE_EMBEDDINGS_IN_MEMORY = True
USE_RERANKING = True
```

**Performance:**
- Model load: 5-10 seconds
- Query response: 1-2 seconds
- Memory usage: 12-16 GB

### Scenario 3: Production Server (GPU cluster)

**Goal:** Maximum throughput and quality

```python
# src/config.py

# Enterprise-grade embedding
EMBEDDING_MODEL = "sentence-transformers/BGE-base-en-v1.5"
EMBEDDING_DEVICE = "cuda"
EMBEDDING_BATCH_SIZE = 128  # Very large batches

# Powerful LLM
LLM_MODEL = "llama2"  # Larger, better quality (7B)
# Or: "dolphin-mixtral:8x7b" for even better quality

# Maximize context and accuracy
TOP_K_CHUNKS = 10
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400

# Production settings
LLM_TEMPERATURE = 0.2  # More deterministic
LLM_MAX_TOKENS = 600
LLM_TIMEOUT = 180

# Enable all optimizations
CACHE_EMBEDDINGS_IN_MEMORY = True
USE_RERANKING = True
NUM_WORKERS = 8

# Other optimizations
USE_GPU = True
VECTOR_DB_TYPE = "faiss"  # Fast index
```

**Performance:**
- Throughput: 10+ queries/second
- Average latency: 500-800ms
- Quality: Highest

---

## 🚀 Speed Optimization

### 1. Use Quantized Models

Quantization reduces model size (3x-4x) with minimal quality loss:

```bash
# Pull quantized models
ollama pull mistral:7b-instruct-q4_0    # 4-bit quantization
ollama pull neural-chat:7b-v3.1-q4_0
ollama pull phi:2.7b-q4_0
```

**Speed improvement:** 2-3x faster
**Quality loss:** < 5%
**Memory:** 1/3 original size

### 2. Reduce Context Size

```python
# src/config.py

# Fewer chunks = faster LLM processing
TOP_K_CHUNKS = 3  # Instead of 5

# Smaller chunks = faster embedding
CHUNK_SIZE = 500  # Instead of 1000

# Faster processing at cost of some context
CHUNK_OVERLAP = 50  # Instead of 200
```

**Speed improvement:** 30-50% faster queries
**Quality impact:** Slightly reduced accuracy

### 3. Batch Inference

For multiple questions, process together:

```python
# Slower: Process one at a time
for question in questions:
    response = chatbot.ask(question)

# Faster: Batch similar questions
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(questions, batch_size=32)  # Vectorized
```

**Speed improvement:** 2-5x for batch processing

### 4. Caching

```python
# Add simple cache
from functools import lru_cache

@lru_cache(maxsize=100)
def answer_cached(question):
    return chatbot.ask(question)

# Same questions answered instantly
answer = answer_cached("What is your return policy?")
answer = answer_cached("What is your return policy?")  # Cached!
```

**Speed improvement:** Instant for repeated questions

### 5. Async Processing

For web services, use async:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

async def ask_async(question):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, chatbot.ask, question)

# Process multiple in parallel
results = await asyncio.gather(
    ask_async("Q1"),
    ask_async("Q2"),
    ask_async("Q3")
)
```

**Speed improvement:** 3-4x for parallel requests

---

## 💾 Memory Optimization

### 1. Smaller Embedding Dimension

Trade embedding quality for memory:

```python
# Current: 384 dimensions
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Smaller: 96 dimensions
# Use quantized version or different model
# This requires reindexing
```

**Memory saved:** 30-40% for vector DB

### 2. Reduce Vector DB Size

```python
# Only keep recent/relevant documents
documents = [d for d in all_documents if is_relevant(d)]

# Lower chunking overlap
CHUNK_OVERLAP = 50  # Instead of 200

# Fewer chunks per document
CHUNK_SIZE = 2000  # Larger chunks = fewer total chunks
```

**Memory saved:** Up to 60%

### 3. Disable Memory Caching

```python
# src/config.py
CACHE_EMBEDDINGS_IN_MEMORY = False
```

**Memory saved:** 50% (tradeoff: slower search)

### 4. Use Disk-Based Vector DB

For very large knowledge bases (1M+ chunks):

```python
# src/config.py
VECTOR_DB_TYPE = "faiss"  # Efficient, disk-friendly

# Use FAISS GPU index for speed
# Or use ChromaDB for flexibility
```

---

## 🎯 Accuracy Optimization

### 1. Better Embedding Model

For more semantic understanding:

```python
# src/config.py
# Current (good, fast)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Better quality (slower)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Best for retrieval (specialized)
EMBEDDING_MODEL = "sentence-transformers/BGE-base-en-v1.5"
```

**Accuracy improvement:** 5-10%
**Speed impact:** 2-3x slower

### 2. Increase Context

```python
# src/config.py
TOP_K_CHUNKS = 10  # Instead of 5
CHUNK_SIZE = 1500  # Larger context
CHUNK_OVERLAP = 400  # More overlap
```

**Accuracy improvement:** 10-15%
**Speed impact:** 2-3x slower

### 3. Lower Relevance Threshold

```python
# src/config.py
RELEVANCE_THRESHOLD = 0.3  # Instead of 0.5
```

**Accuracy improvement:** Capture more relevant context
**Risk:** May include slightly less relevant chunks

### 4. Better LLM

```python
# src/config.py
LLM_MODEL = "llama2"  # Better quality (7B)
# Or
LLM_MODEL = "dolphin-mixtral"  # Very high quality (56B)
```

**Accuracy improvement:** 10-20%
**Speed impact:** Much slower, more memory

### 5. Improve Knowledge Base

**Best ROI for accuracy:**

1. **Remove duplicate information** (consolidate FAQs)
2. **Organize by topic** (separate folders per category)
3. **Add more examples** (how-to guides, use cases)
4. **Update with real customer queries** (add FAQ entries)
5. **Improve document formatting** (clear structure, headers)

```python
# Better document structure example:
"""
RETURN POLICY

Timeframe:
- Standard returns: 30 days from purchase
- Defective items: 90 days from purchase
- Final sale items: No returns

Conditions:
- Items must be in original condition
- Tags and labels must be attached
- Original receipt or proof required

Process:
1. Visit returns.company.com
2. Enter order number
3. Print prepaid shipping label
4. Send item back
5. Receive refund within 5-7 business days
"""
```

**Accuracy improvement:** 15-30%
**Effort:** Medium

### 6. Fine-tune Prompts

```python
# src/config.py

# More specific system prompt
RAG_SYSTEM_PROMPT = """You are a professional customer support agent
for a retail company. Your job is to answer customer questions
accurately and helpfully.

Key responsibilities:
- Answer ONLY using provided context
- Be concise (max 3 sentences)
- If unsure, say 'I don't have this information'
- Be friendly but professional
- Never make exceptions or override policies
"""

# Better query template
QUERY_TEMPLATE = """Based on the company knowledge base below,
answer the customer's question. Be specific and helpful.

Knowledge Base:
{context}

Customer Question: {question}

Your Response (be specific, reference the KB):"""
```

**Accuracy improvement:** 5-10%

---

## 🧠 Advanced Optimizations

### 1. Re-ranking

Use a smaller model to re-rank top-K results:

```python
# src/config.py
USE_RERANKING = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Retrieve 20, rerank to top-5
TOP_K_CHUNKS = 5
INITIAL_RETRIEVE = 20  # Custom config needed
```

**Accuracy:** 10-20% improvement
**Speed:** 20-30% slower

### 2. Hybrid Search

Combine vector search with BM25 (keyword search):

```python
# Requires additional indexing
# Vector search: semantic similarity
# BM25 search: keyword relevance
# Combined: best of both worlds

# Needs custom implementation
```

**Accuracy:** 15-25% improvement

### 3. Query Expansion

Expand queries with synonyms and related terms:

```python
# Before: "What's your shipping?"
# After: "What is your shipping? When will package arrive?
#         How long does delivery take? Shipping time?"

# Needs NLP library (textblob, spacy)
```

**Accuracy:** 5-10% improvement

### 4. Few-Shot Learning

Provide examples in prompt:

```python
QUERY_TEMPLATE = """Examples of good answers:
Q: "How long does shipping take?"
A: "Shipping takes 5-7 business days for standard, 
    2-3 days for express, 1 day for overnight."

Q: "Do you accept returns?"
A: "Yes, within 30 days in original condition."

Now answer this:
{context}
Q: {question}
A:"""
```

**Accuracy:** 10-15% improvement

---

## 📊 Benchmarking

### Create Benchmark Suite

```python
import time
import json
from pathlib import Path

def benchmark_chatbot(chatbot, questions_file):
    """Benchmark chatbot on test questions"""
    
    # Load test questions
    with open(questions_file) as f:
        test_data = json.load(f)
    
    results = []
    total_time = 0
    
    for item in test_data:
        question = item['question']
        
        # Measure time
        start = time.time()
        response = chatbot.ask(question)
        elapsed = time.time() - start
        
        # Record result
        result = {
            'question': question,
            'time': elapsed,
            'confidence': response.get('confidence'),
            'chunks': response.get('context_chunks', 0)
        }
        results.append(result)
        total_time += elapsed
    
    # Calculate metrics
    avg_time = total_time / len(results)
    queries_per_sec = len(results) / total_time
    
    print(f"Queries: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Avg per query: {avg_time:.2f}s")
    print(f"Throughput: {queries_per_sec:.1f} q/s")
    
    return results

# Run benchmark
benchmark_chatbot(chatbot, "test_questions.json")
```

### Comparison Table

| Configuration | Speed | Accuracy | Memory | Best For |
|---|---|---|---|---|
| Phi + MiniLM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Laptop |
| Mistral + MiniLM | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Desktop |
| LLaMA + MPNet | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quality |
| Mixtral + BGE | ⭐⭐ | ⭐⭐⭐⭐⭐⭐ | ⭐⭐ | Production |

---

## 🎓 Tips & Tricks

### Faster First Query

LLM model stays in memory after first query:

```python
# Warm up the model with dummy query
chatbot.ask("Hi")  # ~3 seconds (first time)

# Now real queries are faster
response = chatbot.ask("Your actual question")  # ~1 second
```

### Find Optimal Parameters

```python
# Test different configurations
configs = [
    {'top_k': 3, 'threshold': 0.5},
    {'top_k': 5, 'threshold': 0.5},
    {'top_k': 5, 'threshold': 0.3},
    {'top_k': 7, 'threshold': 0.4},
]

for config in configs:
    chatbot.query_processor.set_parameters(
        top_k=config['top_k'],
        relevance_threshold=config['threshold']
    )
    # Test performance...
```

### Monitor Resource Usage

```bash
# Windows
wmic os get totalvisiblememorysizze
tasklist /v /o csv | grep python

# macOS
memory_stats

# Linux
free -h
top -p $(pgrep -f "python.*app.py")
```

---

**Start with defaults, then optimize based on your constraints!**
