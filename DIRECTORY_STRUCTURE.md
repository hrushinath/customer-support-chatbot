# 📁 Complete Project Directory Structure

```
customer-support-chatbot/
│
├── 📄 README.md                         ⭐ START HERE - Project overview
├── 📄 QUICKSTART.md                     🚀 5-minute setup guide  
├── 📄 PROJECT_SUMMARY.md                📋 This comprehensive summary
├── 📄 ARCHITECTURE.md                   🏗️  System design & RAG explained
├── 📄 SETUP_GUIDE.md                    📖 Detailed installation guide
├── 📄 USAGE_GUIDE.md                    💻 CLI, API, Python usage
├── 📄 OPTIMIZATION.md                   ⚡ Performance tuning
├── 📄 TROUBLESHOOTING.md                🔧 Common issues & fixes
├── 📄 LICENSE.md                        📜 MIT License
├── 📄 requirements.txt                  📦 Python dependencies
├── 📄 test_setup.py                     ✅ Verify installation
│
├── 📂 src/                              🐍 SOURCE CODE
│   ├── __init__.py
│   ├── config.py                        ⚙️  All configuration settings
│   ├── app.py                           🎯 Main application (run this!)
│   │
│   ├── 📂 modules/                      📦 Core Components
│   │   ├── __init__.py
│   │   ├── document_loader.py           📄 Load PDF, DOCX, TXT, JSON
│   │   ├── text_chunker.py              ✂️  Smart text chunking
│   │   ├── embeddings.py                🧠 Sentence embeddings
│   │   ├── vector_database.py           💾 FAISS vector store
│   │   ├── query_processor.py           🔍 Retrieve relevant chunks
│   │   └── response_generator.py        🤖 LLM response generation
│   │
│   └── 📂 utils/                        🛠️  Utilities
│       ├── __init__.py
│       ├── logger.py                    📝 Logging configuration
│       └── helpers.py                   🔧 Helper functions
│
├── 📂 knowledge_base/                   📚 YOUR DOCUMENTS GO HERE
│   ├── 📂 faqs/                         ❓ FAQ Data
│   │   └── general_faq.json             (Sample FAQ file)
│   │
│   └── 📂 documents/                    📄 Documents
│       └── support_documentation.txt    (Sample documentation)
│
├── 📂 vector_store/                     💾 AUTO-GENERATED (don't edit)
│   ├── chatbot_faiss.index              (FAISS vector index)
│   └── chunks_metadata.json             (Chunk metadata)
│
└── 📂 logs/                             📊 AUTO-GENERATED
    └── chatbot.log                      (Application logs)
```

---

## 📖 File Descriptions

### 📄 Documentation Files

| File | Purpose | When to Read |
|------|---------|--------------|
| **README.md** | Project overview, features, quick intro | First time |
| **QUICKSTART.md** | Get started in 5 minutes | When setting up |
| **PROJECT_SUMMARY.md** | Complete project guide | For full understanding |
| **ARCHITECTURE.md** | How RAG works, system design | To understand concepts |
| **SETUP_GUIDE.md** | Detailed step-by-step setup | Troubleshooting setup |
| **USAGE_GUIDE.md** | How to use (CLI, API, code) | After setup complete |
| **OPTIMIZATION.md** | Performance tuning tips | When optimizing |
| **TROUBLESHOOTING.md** | Common problems & fixes | When something breaks |

### 🐍 Source Code Files

| File | Lines | Purpose | Modify? |
|------|-------|---------|---------|
| **src/config.py** | ~200 | All settings & configuration | ✅ YES |
| **src/app.py** | ~400 | Main application logic | ⚠️ Advanced |
| **modules/document_loader.py** | ~250 | Load documents from files | ⚠️ Advanced |
| **modules/text_chunker.py** | ~200 | Split text into chunks | ⚠️ Advanced |
| **modules/embeddings.py** | ~200 | Generate vector embeddings | ❌ Usually no |
| **modules/vector_database.py** | ~300 | FAISS vector store | ❌ Usually no |
| **modules/query_processor.py** | ~200 | Query & retrieve chunks | ⚠️ Advanced |
| **modules/response_generator.py** | ~250 | Generate LLM responses | ⚠️ Advanced |
| **utils/logger.py** | ~50 | Logging setup | ❌ Usually no |
| **utils/helpers.py** | ~80 | Utility functions | ✅ YES |

**Legend:**
- ✅ Safe to modify - configuration and customization
- ⚠️ Advanced - modify if you understand the code
- ❌ Usually no - core functionality, rarely needs changes

### 📚 Knowledge Base

```
knowledge_base/
├── faqs/              # Add your FAQ files here
│   ├── *.json         # Structured FAQ data
│   └── *.txt          # Plain text FAQs
│
└── documents/         # Add your documents here
    ├── *.pdf          # PDF documents
    ├── *.docx         # Word documents
    ├── *.txt          # Text files
    └── *.md           # Markdown files
```

**Supported formats:**
- ✅ JSON (`.json`) - Structured data
- ✅ Text (`.txt`, `.md`) - Plain text
- ✅ PDF (`.pdf`) - Requires PyPDF2
- ✅ Word (`.docx`) - Requires python-docx

**To add documents:**
1. Copy files to appropriate folder
2. Restart app: `python src/app.py`
3. System automatically rebuilds index

---

## 🎯 Where to Start

### First Time Users
```
1. README.md          (5 min)  - Understand what this is
2. QUICKSTART.md      (10 min) - Set it up
3. Try the chatbot    (5 min)  - Ask questions
4. ARCHITECTURE.md    (20 min) - Learn how it works
```

### Developers
```
1. PROJECT_SUMMARY.md (10 min) - Complete overview
2. src/config.py      (5 min)  - Configuration options
3. src/app.py         (10 min) - Main application flow
4. modules/*.py       (30 min) - Core components
```

### Customizers
```
1. USAGE_GUIDE.md     (15 min) - Learn all usage options
2. src/config.py      (10 min) - Tweak settings
3. knowledge_base/    (-)      - Add your documents
4. OPTIMIZATION.md    (20 min) - Tune performance
```

---

## 🔍 Quick Reference

### Run the Chatbot
```bash
python src/app.py
```

### Test Setup
```bash
python test_setup.py
```

### Change Models
```bash
# Pull different model
ollama pull neural-chat

# Edit config
# src/config.py → LLM_MODEL = "neural-chat"
```

### Add Documents
```bash
# 1. Copy files
cp my_faq.json knowledge_base/faqs/
cp my_doc.pdf knowledge_base/documents/

# 2. Restart app (auto-rebuilds)
python src/app.py
```

### View Logs
```bash
# Real-time logs
tail -f logs/chatbot.log

# All logs
cat logs/chatbot.log
```

### Reset Everything
```bash
# Delete generated data
rm -rf vector_store/ logs/

# Restart (rebuilds from scratch)
python src/app.py
```

---

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | ~25 files |
| **Lines of Code** | ~2,500 lines |
| **Documentation** | ~8,000 words |
| **Setup Time** | 5-10 minutes |
| **Models Size** | 3-15 GB (depends on choice) |
| **Memory Usage** | 4-16 GB (depends on config) |

---

## 🎓 Learning Path

### Beginner (Week 1)
- Day 1: Setup + Try basic queries
- Day 2: Read ARCHITECTURE.md
- Day 3: Add your own documents
- Day 4: Experiment with config
- Day 5: Deploy as web app

### Intermediate (Week 2-3)
- Week 2: Optimize performance
- Week 3: Build REST API, integrate

### Advanced (Month 2)
- Add custom features
- Implement re-ranking
- Fine-tune for production
- Scale to multiple users

---

## 💡 Pro Tips

### Speed up Development
```python
# Use smaller model during dev
LLM_MODEL = "phi"  # Fast, 3.8B

# Fewer chunks = faster
TOP_K_CHUNKS = 3

# Small batch size
EMBEDDING_BATCH_SIZE = 8
```

### Improve Accuracy
```python
# Better models
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_MODEL = "llama2"

# More context
TOP_K_CHUNKS = 7
CHUNK_SIZE = 1200

# Lower temperature (more deterministic)
LLM_TEMPERATURE = 0.2
```

### Save Memory
```python
# Lightweight model
LLM_MODEL = "phi"

# Disable caching
CACHE_EMBEDDINGS_IN_MEMORY = False

# Smaller batches
EMBEDDING_BATCH_SIZE = 8
```

---

## 🚀 Quick Commands

```bash
# Setup
pip install -r requirements.txt
ollama serve
ollama pull mistral

# Run
python src/app.py

# Test
python test_setup.py

# Add docs & rebuild
cp my_docs/* knowledge_base/documents/
python src/app.py

# Reset
rm -rf vector_store/ logs/
python src/app.py

# Check logs
tail -f logs/chatbot.log
```

---

## 📞 Need Help?

**Order of troubleshooting:**
1. Check TROUBLESHOOTING.md
2. Run `python test_setup.py`
3. Check logs: `logs/chatbot.log`
4. Verify Ollama: `ollama list`
5. Reset and rebuild

**Common issues:**
- Ollama not running → `ollama serve`
- Model missing → `ollama pull mistral`
- Out of memory → Use `phi` model
- Slow → Use quantized model

---

**Everything you need to build a production-ready RAG chatbot! 🚀**
