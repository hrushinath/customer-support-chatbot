# 🚀 Quick Start Guide

Welcome! This guide will get you from zero to chatbot in **5 minutes**.

## ✅ Before You Start

Make sure you have:
- [ ] Python 3.8+ installed (`python --version`)
- [ ] Internet connection (for first-time downloads only)
- [ ] 8GB RAM minimum (16GB recommended)
- [ ] 10GB free disk space (for models)

---

## 📝 Step-by-Step Setup

### Step 1: Install Python Dependencies (2 minutes)

Open terminal/command prompt in the project folder:

```bash
# Install all required packages
pip install -r requirements.txt
```

**This will install:**
- sentence-transformers (for embeddings)
- faiss-cpu (for vector database)
- numpy (for numerical operations)
- PyPDF2 (for PDF handling)
- python-docx (for Word documents)

---

### Step 2: Install Ollama (2 minutes)

**Download Ollama:**
1. Go to https://ollama.ai
2. Download for your OS (Windows/Mac/Linux)
3. Install (just click through the installer)

**Start Ollama:**
```bash
# In a terminal, run (keep this terminal open):
ollama serve
```

You should see: "Ollama is running"

---

### Step 3: Download Language Model (1 minute setup, ~5 minute download)

**In another terminal:**

```bash
# Download Mistral 7B (recommended, ~4GB)
ollama pull mistral
```

**Alternative models:**
```bash
# Lightweight option for laptops:
ollama pull phi

# Or chat-optimized:
ollama pull neural-chat
```

The download happens once. Models stay on your computer.

---

### Step 4: Run the Chatbot! (Instant)

```bash
python src/app.py
```

**First run will:**
1. Download embedding model (~400MB, one-time)
2. Load sample knowledge base
3. Generate embeddings (takes ~10-30 seconds)
4. Start interactive chat

**Subsequent runs:** Start instantly (< 1 second)!

---

## 💬 Try These Questions

Once chatbot starts, try:

```
You: What is your return policy?

You: How long does shipping take?

You: Do you ship internationally?

You: What payment methods do you accept?
```

---

## 🎯 What's Next?

### 1. Add Your Own Knowledge Base

Replace sample documents with yours:

```bash
# Add to these folders:
knowledge_base/
  ├── faqs/           # JSON, TXT files
  └── documents/      # PDF, DOCX, TXT files
```

Then restart the chatbot - it auto-rebuilds!

### 2. Customize Configuration

Edit `src/config.py`:

```python
# Change LLM model
LLM_MODEL = "mistral"  # or "phi", "llama2", etc.

# Adjust retrieval
TOP_K_CHUNKS = 5       # How many chunks to use
RELEVANCE_THRESHOLD = 0.5  # Minimum relevance

# Tune LLM
LLM_TEMPERATURE = 0.3  # 0=strict, 1=creative
```

### 3. Create a Web Interface

**Option A: Simple Web UI (Streamlit)**
```bash
pip install streamlit
streamlit run examples/streamlit_app.py
```

**Option B: REST API (FastAPI)**
```bash
pip install fastapi uvicorn
python examples/api_server.py
```

---

## 🆘 Troubleshooting

### "Cannot connect to Ollama"
**Fix:** Make sure Ollama is running: `ollama serve`

### "Model 'mistral' not found"
**Fix:** Pull the model: `ollama pull mistral`

### "Out of memory"
**Fix 1:** Use smaller model: `ollama pull phi`
**Fix 2:** Close other apps
**Fix 3:** In `src/config.py`, set `LLM_MODEL = "phi"`

### "No documents found"
**Fix:** Add files to `knowledge_base/faqs/` or `knowledge_base/documents/`

### Other issues?
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed help.

---

## 📚 Learn More

- **[README.md](README.md)** - Project overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - How RAG works (beginner-friendly)
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Advanced usage (API, Python, etc.)
- **[OPTIMIZATION.md](OPTIMIZATION.md)** - Performance tuning

---

## ✅ Verify Setup

Run the test script:

```bash
python test_setup.py
```

This checks:
- ✓ All Python packages installed
- ✓ Project files present
- ✓ Ollama running
- ✓ Configuration valid

---

## 🎉 Success!

If you see the interactive chat interface, you're all set!

**Type your questions and get answers from your knowledge base.**

Questions? Check the documentation or troubleshooting guide.

---

## 📊 Performance Expectations

| Setup | First Query | Later Queries | Memory |
|-------|-------------|---------------|--------|
| Laptop (8GB, Phi) | ~2s | ~0.5s | 4-6 GB |
| Desktop (16GB, Mistral) | ~3s | ~1s | 12-14 GB |
| Server (GPU, LLaMA) | ~1s | ~0.3s | 16+ GB |

**Note:** First query includes model loading. Subsequent queries are much faster!

---

## 🚀 Ready?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (in one terminal)
ollama serve

# 3. Pull model (in another terminal)
ollama pull mistral

# 4. Run chatbot
python src/app.py

# That's it! 🎉
```

---

**Built with ❤️ for local, private, and accurate AI**

*Questions? See the docs or TROUBLESHOOTING.md*
