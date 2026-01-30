# Setup and Installation Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.8+ (Python 3.10+ recommended)
- 8GB RAM minimum (16GB recommended for optimal performance)
- Windows, macOS, or Linux
- Internet connection (only for first-time model downloads)

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd customer-support-chatbot

# Install required packages
pip install -r requirements.txt
```

**Expected installation time:** 2-5 minutes

### Step 2: Install Ollama

Ollama lets you run large language models locally.

1. **Download Ollama** from https://ollama.ai
2. **Install** for your operating system (Windows/macOS/Linux)
3. **Start Ollama server**:
   ```bash
   ollama serve
   ```
   (Keep this terminal open in the background)

4. **Pull a model** (in another terminal):
   ```bash
   # Recommended: Fast and good quality
   ollama pull mistral
   
   # Or other options:
   ollama pull neural-chat    # Optimized for chat
   ollama pull phi            # Lightweight (3.8B)
   ollama pull llama2         # Larger (7B)
   ```

**First pull downloads 3-15GB (depending on model)** - this is one-time only

### Step 3: Verify Installation

```bash
# Check Python packages
python -c "import sentence_transformers, faiss, numpy; print('✓ All packages installed')"

# Check Ollama is running
curl http://localhost:11434/api/tags

# You should see JSON output with available models
```

### Step 4: Run the Chatbot

```bash
python src/app.py
```

**First run will:**
1. Download embedding model (~400MB) - one-time only
2. Load documents from knowledge_base/
3. Generate embeddings for all chunks
4. Build vector index (FAISS)
5. Start interactive chat

**Subsequent runs:** Instant startup (< 1 second)

---

## 📋 Detailed Installation Steps

### For Windows Users

```powershell
# 1. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download Ollama from https://ollama.ai and install
# 4. Open PowerShell as Administrator and start Ollama
ollama serve

# In another PowerShell window:
# 5. Pull model
ollama pull mistral

# 6. Run chatbot
python src\app.py
```

### For macOS Users

```bash
# 1. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama via Homebrew
brew install ollama

# 5. Start Ollama
ollama serve

# In another terminal:
# 6. Pull model
ollama pull mistral

# 7. Run chatbot
python src/app.py
```

### For Linux Users (Ubuntu/Debian)

```bash
# 1. Install Python dev tools
sudo apt-get update
sudo apt-get install python3-dev python3-pip python3-venv

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama
curl https://ollama.ai/install.sh | sh

# 5. Start Ollama
ollama serve

# In another terminal:
# 6. Pull model
ollama pull mistral

# 7. Run chatbot
python src/app.py
```

### For Linux Users (Alternative - Docker)

```bash
# If you prefer Docker:
docker run -d -p 11434:11434 ollama/ollama

# Then pull model:
docker exec -it <container_id> ollama pull mistral
```

---

## 🔧 Configuration

All settings are in `src/config.py`. Key options:

```python
# Embedding Model (language understanding)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model (answer generation) 
LLM_MODEL = "mistral"  # Change to "phi" for lightweight, "llama2" for larger

# Retrieval settings
TOP_K_CHUNKS = 5  # Return top-5 relevant chunks
RELEVANCE_THRESHOLD = 0.5  # Minimum relevance score

# Chunk size (larger = more context but slower)
CHUNK_SIZE = 1000  # characters (~250 tokens)
CHUNK_OVERLAP = 200

# LLM Parameters
LLM_TEMPERATURE = 0.3  # 0=deterministic, 1=creative
LLM_MAX_TOKENS = 500  # Max response length

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
```

**After changing config, restart the chatbot and it will rebuild indexes.**

---

## 📦 Virtual Environment (Recommended)

To avoid conflicts with other Python packages:

### Create virtual environment:
```bash
python -m venv venv
```

### Activate it:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

### Install packages in virtual environment:
```bash
pip install -r requirements.txt
```

### Deactivate when done:
```bash
deactivate
```

---

## 🚨 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'sentence_transformers'"
**Solution:**
```bash
pip install sentence-transformers
```

### Problem: "Cannot connect to Ollama"
**Solution:**
1. Make sure Ollama is running: `ollama serve`
2. Check it's on the right port: http://localhost:11434/api/tags
3. Verify firewall isn't blocking port 11434

### Problem: "Model 'mistral' not found"
**Solution:**
```bash
ollama pull mistral
# Wait for download to complete (3-4GB)
```

### Problem: "Out of memory" errors
**Solution:**
1. Use smaller model: `ollama pull phi` (3.8B)
2. Reduce batch size in config.py: `EMBEDDING_BATCH_SIZE = 16`
3. Close other applications
4. Check available RAM: `free -h` (Linux) or Task Manager (Windows)

### Problem: "FAISS index file not found"
**Solution:**
- Delete `vector_store/` folder
- Restart app - it will rebuild the index automatically

### Problem: "Slow response times"
**Solution:**
1. Use quantized model: `ollama pull mistral:7b-instruct-q4_0`
2. Reduce `TOP_K_CHUNKS` in config.py (e.g., 3 instead of 5)
3. Check CPU usage - ensure no other heavy processes

---

## ⚡ Performance Tips

### For Laptops (8GB RAM, CPU only):
```python
# src/config.py
LLM_MODEL = "phi"  # Smaller model
EMBEDDING_BATCH_SIZE = 16
TOP_K_CHUNKS = 3
LLM_TEMPERATURE = 0.2
```

### For Desktops (16GB+ RAM, GPU available):
```python
# src/config.py
LLM_MODEL = "mistral"  # Good balance
EMBEDDING_BATCH_SIZE = 64
TOP_K_CHUNKS = 5
USE_GPU = True  # If GPU available
```

### For High-Quality Answers (Time not critical):
```python
# src/config.py
LLM_MODEL = "llama2"  # Better quality
TOP_K_CHUNKS = 10  # More context
LLM_TEMPERATURE = 0.2  # Less creative, more accurate
```

---

## 📚 Adding Your Knowledge Base

Replace sample documents with your own:

1. **Add to `knowledge_base/faqs/`**:
   - JSON files (FAQ data)
   - Text files (documentation)

2. **Add to `knowledge_base/documents/`**:
   - PDF files (manuals, policies)
   - Word documents (.docx)
   - Text files (.txt)
   - Markdown files (.md)

3. **Restart app**:
   ```bash
   python src/app.py
   ```
   It will automatically rebuild the vector index with your new documents.

---

## 🎯 Next Steps

1. ✅ Complete setup above
2. 📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) for system explanation
3. 💬 Try running the app: `python src/app.py`
4. 🧪 Ask test questions in interactive mode
5. 📝 Add your own knowledge base documents
6. 🎨 (Optional) Create a web UI with Streamlit or FastAPI

---

## 📞 Support

If you encounter issues:

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review logs in `logs/chatbot.log`
3. Check Ollama is running: `ollama serve`
4. Verify Python version: `python --version` (should be 3.8+)

---

**Ready to go?** Run `python src/app.py` and start chatting! 🚀
