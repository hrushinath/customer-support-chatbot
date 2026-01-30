"""
Simple test file to verify chatbot setup
Run: python test_setup.py
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("=" * 70)
    print("TESTING IMPORTS")
    print("=" * 70)
    
    packages = {
        'numpy': 'NumPy',
        'sentence_transformers': 'Sentence-Transformers',
        'faiss': 'FAISS',
        'PyPDF2': 'PyPDF2',
        'docx': 'python-docx',
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            print(f"  Install with: pip install {package}")
            all_good = False
    
    return all_good


def test_directories():
    """Test if project directories exist"""
    print("\n" + "=" * 70)
    print("TESTING DIRECTORIES")
    print("=" * 70)
    
    dirs_to_check = [
        'src',
        'src/modules',
        'src/utils',
        'knowledge_base',
        'knowledge_base/faqs',
        'knowledge_base/documents',
        'vector_store',
        'logs'
    ]
    
    all_good = True
    for dir_name in dirs_to_check:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ - NOT FOUND")
            all_good = False
    
    return all_good


def test_files():
    """Test if critical files exist"""
    print("\n" + "=" * 70)
    print("TESTING FILES")
    print("=" * 70)
    
    files_to_check = [
        'src/config.py',
        'src/app.py',
        'src/modules/document_loader.py',
        'src/modules/embeddings.py',
        'src/modules/vector_database.py',
        'src/modules/query_processor.py',
        'src/modules/response_generator.py',
        'requirements.txt',
        'README.md',
        'ARCHITECTURE.md',
        'SETUP_GUIDE.md'
    ]
    
    all_good = True
    for file_name in files_to_check:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - NOT FOUND")
            all_good = False
    
    return all_good


def test_knowledge_base():
    """Test if knowledge base has documents"""
    print("\n" + "=" * 70)
    print("TESTING KNOWLEDGE BASE")
    print("=" * 70)
    
    kb_dir = Path('knowledge_base')
    
    # Count documents
    faq_files = list(kb_dir.glob('faqs/*'))
    doc_files = list(kb_dir.glob('documents/*'))
    
    print(f"FAQ files: {len(faq_files)}")
    for f in faq_files:
        print(f"  - {f.name}")
    
    print(f"\nDocument files: {len(doc_files)}")
    for f in doc_files:
        print(f"  - {f.name}")
    
    has_docs = len(faq_files) > 0 or len(doc_files) > 0
    if has_docs:
        print(f"\n✓ Knowledge base has documents")
    else:
        print(f"\n⚠ Knowledge base is empty - add documents to knowledge_base/")
    
    return has_docs


def test_ollama():
    """Test if Ollama is running"""
    print("\n" + "=" * 70)
    print("TESTING OLLAMA CONNECTION")
    print("=" * 70)
    
    try:
        import urllib.request
        import json
        
        url = "http://localhost:11434/api/tags"
        request = urllib.request.Request(url, method='GET')
        
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            
            if 'models' in data:
                models = data['models']
                print(f"✓ Ollama is running!")
                print(f"Available models: {len(models)}")
                
                for model in models:
                    name = model.get('name', 'unknown')
                    print(f"  - {name}")
                
                if len(models) == 0:
                    print("\n⚠ No models pulled yet")
                    print("Run: ollama pull mistral")
                    return False
                
                return True
        
    except Exception as e:
        print(f"✗ Cannot connect to Ollama")
        print(f"Error: {str(e)}")
        print(f"\nMake sure:")
        print(f"1. Ollama is installed from https://ollama.ai")
        print(f"2. Ollama is running: ollama serve")
        print(f"3. A model is pulled: ollama pull mistral")
        return False


def test_config():
    """Test configuration file"""
    print("\n" + "=" * 70)
    print("TESTING CONFIGURATION")
    print("=" * 70)
    
    try:
        from src import config
        
        print(f"✓ Configuration loaded")
        print(f"\nKey settings:")
        print(f"  Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"  LLM Model: {config.LLM_MODEL}")
        print(f"  Ollama URL: {config.OLLAMA_BASE_URL}")
        print(f"  Top-K Chunks: {config.TOP_K_CHUNKS}")
        print(f"  Chunk Size: {config.CHUNK_SIZE}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading configuration: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " CUSTOMER SUPPORT CHATBOT - SETUP TEST ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    results = {
        'Imports': test_imports(),
        'Directories': test_directories(),
        'Files': test_files(),
        'Knowledge Base': test_knowledge_base(),
        'Configuration': test_config(),
        'Ollama': test_ollama(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to run!")
        print("\nStart chatbot with:")
        print("  python src/app.py")
    else:
        print("✗ SOME TESTS FAILED - Fix issues above and try again")
        print("\nFor help, see:")
        print("  - SETUP_GUIDE.md")
        print("  - TROUBLESHOOTING.md")
    
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
