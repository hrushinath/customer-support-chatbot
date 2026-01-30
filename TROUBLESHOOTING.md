# Troubleshooting Guide

## 🔧 Common Issues and Solutions

### Installation Issues

#### Problem: `ModuleNotFoundError: No module named 'sentence_transformers'`

**Cause:** Python packages not installed

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sentence_transformers; print('✓ OK')"
```

---

#### Problem: `pip install faiss-cpu` fails

**Cause:** Compatibility issue or missing build tools

**Solution - Windows:**
```bash
# Try pre-built wheel
pip install faiss-cpu --no-build-isolation

# Or use CPU version specifically
pip install faiss-cpu
```

**Solution - macOS:**
```bash
# Install build tools first
brew install cmake

# Then install
pip install faiss-cpu
```

**Solution - Linux:**
```bash
# Install dev tools
sudo apt-get install python3-dev build-essential

# Then install
pip install faiss-cpu
```

**Alternative:** Use conda
```bash
conda install -c conda-forge faiss-cpu
```

---

### Ollama Issues

#### Problem: "Cannot connect to Ollama at http://localhost:11434"

**Cause:** Ollama not running or not accessible

**Solution:**

1. **Make sure Ollama is running:**
   ```bash
   ollama serve
   ```
   Keep this terminal open!

2. **In another terminal, verify connection:**
   ```bash
   curl http://localhost:11434/api/tags
   ```
   Should return JSON with available models.

3. **If still not working:**
   - Check firewall isn't blocking port 11434
   - Try: http://127.0.0.1:11434 instead of localhost
   - Reinstall Ollama from https://ollama.ai

---

#### Problem: "Model 'mistral' not found"

**Cause:** Model not downloaded yet

**Solution:**
```bash
# Pull the model (will download 3-4 GB)
ollama pull mistral

# Wait for download to complete
# Then restart your chatbot
```

**Alternative models:**
```bash
ollama pull phi            # Lightweight (2.7B)
ollama pull neural-chat    # Chat optimized (7B)
ollama pull llama2         # High quality (7B)
```

---

#### Problem: Ollama process takes all memory

**Cause:** Model too large for available RAM

**Solution:**

1. **Use smaller model:**
   ```bash
   ollama pull phi  # Only 3.8B parameters
   ```
   
2. **In src/config.py:**
   ```python
   LLM_MODEL = "phi"
   ```

3. **Use quantized model (smaller, faster):**
   ```bash
   ollama pull mistral:7b-instruct-q4_0  # 4-bit quantized
   ```

4. **Check available memory:**
   - Windows: Task Manager → Performance
   - macOS: Activity Monitor
   - Linux: `free -h`

---

### Chatbot Initialization Issues

#### Problem: "Embedding model download fails"

**Cause:** No internet or network issues

**Solution:**

1. **Check internet connection:**
   ```bash
   ping google.com
   ```

2. **Retry download:**
   ```bash
   # Delete downloaded model
   rm -rf ~/.cache/huggingface/

   # Restart app to re-download
   python src/app.py
   ```

3. **Manual download:**
   ```bash
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   ```

---

#### Problem: "No documents found in knowledge base"

**Cause:** Knowledge base empty or path wrong

**Solution:**

1. **Check knowledge base exists:**
   ```bash
   ls knowledge_base/
   ls knowledge_base/faqs/
   ls knowledge_base/documents/
   ```

2. **Add sample documents:**
   - Download sample PDFs to `knowledge_base/documents/`
   - Or copy text files to `knowledge_base/faqs/`

3. **Check file permissions:**
   ```bash
   chmod 644 knowledge_base/faqs/*.json
   chmod 644 knowledge_base/documents/*.*
   ```

4. **Verify supported formats:**
   - Text: .txt, .md
   - Data: .json
   - Documents: .pdf, .docx

---

#### Problem: "FAISS index corrupted"

**Cause:** Vector store files damaged

**Solution:**

1. **Delete vector store:**
   ```bash
   rm -rf vector_store/
   rm -rf logs/
   ```

2. **Restart app:**
   ```bash
   python src/app.py
   ```
   It will rebuild the index automatically.

---

### Query Processing Issues

#### Problem: "Query takes very long time (> 10s)"

**Cause:** LLM is slow or system overloaded

**Solutions:**

1. **Use smaller/quantized model:**
   ```bash
   ollama pull phi
   # In config: LLM_MODEL = "phi"
   ```

2. **Reduce context size:**
   ```python
   # src/config.py
   TOP_K_CHUNKS = 3  # Instead of 5
   CHUNK_SIZE = 800  # Instead of 1000
   ```

3. **Check CPU/memory usage:**
   ```bash
   # Windows Task Manager
   # macOS Activity Monitor
   # Linux: top
   ```

4. **Close other applications** taking memory/CPU

5. **Check if first query:**
   - First query includes model loading (~3s)
   - Subsequent queries should be faster

---

#### Problem: "Out of memory error"

**Cause:** Not enough RAM for model

**Solutions:**

1. **Use smaller model:**
   ```python
   # src/config.py
   LLM_MODEL = "phi"  # 3.8B (uses ~4GB)
   ```

2. **Reduce embedding batch size:**
   ```python
   # src/config.py
   EMBEDDING_BATCH_SIZE = 8  # Instead of 32
   ```

3. **Disable memory caching:**
   ```python
   # src/config.py
   CACHE_EMBEDDINGS_IN_MEMORY = False
   ```

4. **Check available memory:**
   ```bash
   # Linux
   free -h
   
   # macOS
   vm_stat
   
   # Windows
   systeminfo | grep "Total Physical Memory"
   ```

---

#### Problem: "Inaccurate or off-topic answers"

**Cause:** Poor retrieval or LLM confusion

**Solutions:**

1. **Increase context:**
   ```python
   # src/config.py
   TOP_K_CHUNKS = 7  # More context
   RELEVANCE_THRESHOLD = 0.3  # Lower threshold
   ```

2. **Better embedding model:**
   ```python
   # src/config.py
   EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
   ```

3. **Better LLM:**
   ```bash
   ollama pull llama2
   ```
   ```python
   # src/config.py
   LLM_MODEL = "llama2"
   ```

4. **Improve knowledge base:**
   - Add more documents
   - Remove duplicates
   - Better formatting

5. **Fine-tune prompt:**
   ```python
   # src/config.py
   RAG_SYSTEM_PROMPT = """..."""  # More specific instructions
   ```

---

#### Problem: "Getting same answer for different questions"

**Cause:** Generic response from LLM

**Solutions:**

1. **Increase temperature (more creative):**
   ```python
   # src/config.py
   LLM_TEMPERATURE = 0.5  # Instead of 0.3
   ```

2. **Better retrieval:**
   ```python
   # src/config.py
   TOP_K_CHUNKS = 5  # Check more chunks
   RELEVANCE_THRESHOLD = 0.4  # Balanced
   ```

3. **Diverse knowledge base:**
   - Add varied documentation
   - Include examples and edge cases

---

### Python/Environment Issues

#### Problem: "Python: command not found"

**Cause:** Python not installed or not in PATH

**Solution:**

1. **Install Python:** https://python.org
2. **Verify installation:**
   ```bash
   python --version  # or python3
   pip --version
   ```
3. **Use correct command:**
   - macOS/Linux: `python3`
   - Windows: `python`

---

#### Problem: "pip: command not found"

**Cause:** pip not installed

**Solution:**

```bash
# macOS/Linux
python3 -m pip install --upgrade pip

# Windows
python -m pip install --upgrade pip
```

---

#### Problem: "Virtual environment issues"

**Cause:** Wrong environment active

**Solution:**

```bash
# Create fresh environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

### Port and Network Issues

#### Problem: "Address already in use: ('127.0.0.1', 11434)"

**Cause:** Ollama already running

**Solution:**
- Ollama should run in background
- Just one instance needed
- Kill old process if necessary:
  ```bash
  # Linux/macOS
  pkill ollama
  
  # Windows
  taskkill /F /IM ollama.exe
  ```

---

#### Problem: "Port 8000 already in use" (FastAPI)

**Cause:** Another process using port 8000

**Solution:**

```python
# api_server.py - change port
uvicorn.run(app, host="0.0.0.0", port=8001)
```

Or find and kill existing process:
```bash
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 📋 Debugging Checklist

Before asking for help:

- [ ] Python 3.8+ installed: `python --version`
- [ ] Requirements installed: `pip list | grep sentence`
- [ ] Ollama running: `curl http://localhost:11434/api/tags`
- [ ] Model pulled: `ollama list`
- [ ] Knowledge base exists: `ls knowledge_base/`
- [ ] Can import modules: `python -c "import src.app"`
- [ ] Check logs: `tail -f logs/chatbot.log`

---

## 🐛 Reporting Bugs

When reporting issues, include:

1. **Your setup:**
   ```bash
   python --version
   pip list
   ollama -v
   uname -a  # or systeminfo on Windows
   ```

2. **Error message:**
   - Full stack trace
   - Last few log lines

3. **What you tried:**
   - Steps to reproduce
   - What you expected vs. what happened

4. **Configuration:**
   - Your src/config.py settings
   - Knowledge base size

---

## 📚 Additional Resources

- [Official Ollama Docs](https://github.com/jmorganca/ollama)
- [Sentence-Transformers Docs](https://www.sbert.net)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

## 💬 Getting Help

1. **Check this troubleshooting guide first** ← You are here
2. **Read the documentation** in main README
3. **Check logs:** `logs/chatbot.log`
4. **Search online** for the error message
5. **Ask in discussions** if still stuck

---

**Still stuck?** Try one of these:

1. **Delete and rebuild:**
   ```bash
   rm -rf vector_store/ logs/
   python src/app.py
   ```

2. **Start fresh:**
   ```bash
   # Fresh Python environment
   python -m venv venv_fresh
   source venv_fresh/bin/activate  # or venv_fresh\Scripts\activate on Windows
   pip install -r requirements.txt
   python src/app.py
   ```

3. **Check system resources:**
   ```bash
   # How much free RAM?
   free -h  # Linux
   vm_stat  # macOS
   ```

Good luck! 🚀
