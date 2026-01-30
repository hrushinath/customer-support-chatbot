"""
Customer Support Chatbot - Main Application
Orchestrates all modules for RAG-based question answering
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from modules.document_loader import DocumentLoader
from modules.text_chunker import TextChunker
from modules.embeddings import EmbeddingGenerator
from modules.vector_database import FAISSVectorStore
from modules.query_processor import QueryProcessor
from modules.response_generator import ResponseGenerator
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__, log_file=LOG_FILE)


class CustomerSupportChatbot:
    """
    Main chatbot class that orchestrates all components:
    1. Document Loading
    2. Chunking
    3. Embedding
    4. Vector DB Storage
    5. Query Processing
    6. Response Generation
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialize the chatbot
        
        Args:
            config_path: Optional path to configuration file
            verbose: Print status messages
        """
        self.verbose = verbose
        self._log(f"Initializing CustomerSupportChatbot...")
        
        # Initialize components (lazy loading)
        self.embedding_generator = None
        self.vector_store = None
        self.query_processor = None
        self.response_generator = None
        self.is_initialized = False
        self._log("Chatbot created (not yet initialized)")

    def initialize(self) -> bool:
        """
        Initialize all components and load/build vector database
        
        Returns:
            True if initialization successful
        """
        try:
            self._log("Starting initialization...")
            
            # Step 1: Initialize Embedding Generator
            self._log("Loading embedding model...")
            self.embedding_generator = EmbeddingGenerator(
                model_name=EMBEDDING_MODEL,
                device=EMBEDDING_DEVICE
            )
            
            # Step 2: Initialize Vector Store
            self._log("Initializing vector database...")
            self.vector_store = FAISSVectorStore(
                embedding_dim=self.embedding_generator.dimension
            )
            
            # Step 3: Try to load existing index, otherwise build it
            if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
                self._log("Loading existing vector index...")
                self.vector_store = FAISSVectorStore.load(
                    FAISS_INDEX_PATH,
                    METADATA_PATH
                )
            else:
                self._log("Building new vector index...")
                self._build_vector_index()
            
            # Step 4: Initialize Query Processor
            self.query_processor = QueryProcessor(
                embedding_generator=self.embedding_generator,
                vector_store=self.vector_store,
                top_k=TOP_K_CHUNKS,
                relevance_threshold=RELEVANCE_THRESHOLD
            )
            
            # Step 5: Initialize Response Generator
            self.response_generator = ResponseGenerator(
                ollama_url=OLLAMA_BASE_URL,
                model_name=LLM_MODEL,
                system_prompt=RAG_SYSTEM_PROMPT,
                query_template=QUERY_TEMPLATE,
                temperature=LLM_TEMPERATURE,
                top_p=LLM_TOP_P,
                max_tokens=LLM_MAX_TOKENS,
                timeout=LLM_TIMEOUT
            )
            
            self.is_initialized = True
            self._log(f"✓ Initialization complete!")
            self._log(f"  Vector DB: {len(self.vector_store)} chunks")
            self._log(f"  Embedding Model: {EMBEDDING_MODEL}")
            self._log(f"  LLM Model: {LLM_MODEL}")
            
            return True
        
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            self._log(f"✗ Initialization failed: {str(e)}", error=True)
            return False

    def _build_vector_index(self) -> None:
        """Build vector index from knowledge base documents"""
        # Step 1: Load documents
        self._log("Loading documents from knowledge base...")
        loader = DocumentLoader()
        documents = loader.load_from_directory(KNOWLEDGE_BASE_DIR)
        
        if not documents:
            self._log("⚠ No documents found in knowledge base!", warning=True)
            return
        
        self._log(f"  Loaded {len(documents)} documents")
        
        # Step 2: Chunk documents
        self._log("Chunking documents...")
        chunker = TextChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator=CHUNK_SEPARATOR,
            min_chunk_length=MIN_CHUNK_LENGTH
        )
        chunks = chunker.chunk_documents(documents, verbose=False)
        self._log(f"  Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        self._log("Generating embeddings...")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_generator.embed_texts(
            chunk_texts,
            batch_size=EMBEDDING_BATCH_SIZE
        )
        
        # Step 4: Prepare metadata
        metadata_list = [
            {
                'text': chunk.text,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'start_idx': chunk.start_idx,
                'end_idx': chunk.end_idx,
                'doc_type': chunk.doc_type,
                'length': len(chunk.text)
            }
            for chunk in chunks
        ]
        
        # Step 5: Store in vector DB
        self._log("Storing embeddings in vector database...")
        self.vector_store.add_embeddings(embeddings, metadata_list)
        
        # Step 6: Save to disk
        self._log("Saving vector index to disk...")
        self.vector_store.save(FAISS_INDEX_PATH, METADATA_PATH)
        self._log(f"  Saved to {FAISS_INDEX_PATH}")

    def ask(
        self,
        question: str,
        return_sources: bool = INCLUDE_SOURCES
    ) -> Dict:
        """
        Ask the chatbot a question
        
        Args:
            question: User's question
            return_sources: Include source information
            
        Returns:
            Dict with:
            - question: The user's question
            - answer: The generated answer
            - confidence: How confident the answer is (high/medium/low)
            - sources: Source chunks (if requested)
            - success: Whether the query succeeded
        """
        if not self.is_initialized:
            return {
                'success': False,
                'error': 'Chatbot not initialized. Call initialize() first.'
            }
        
        try:
            logger.info(f"Processing question: {question}")
            
            # Step 1: Get context from vector DB
            context_info = self.query_processor.get_answer_context(
                question,
                include_metadata=return_sources
            )
            
            # Step 2: Generate response
            response = self.response_generator.answer_query(
                question=question,
                context=context_info['context'],
                context_info=context_info
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'success': False,
                'question': question,
                'error': str(e)
            }

    def batch_ask(self, questions: List[str]) -> List[Dict]:
        """
        Ask multiple questions at once
        
        Args:
            questions: List of questions
            
        Returns:
            List of responses
        """
        results = []
        for i, question in enumerate(questions, 1):
            self._log(f"Processing question {i}/{len(questions)}...")
            response = self.ask(question)
            results.append(response)
        
        return results

    def reload_knowledge_base(self) -> bool:
        """Reload and rebuild vector index from knowledge base"""
        try:
            self._log("Reloading knowledge base...")
            self.vector_store.clear()
            self._build_vector_index()
            return True
        except Exception as e:
            self._log(f"Failed to reload: {str(e)}", error=True)
            return False

    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        return {
            'initialized': self.is_initialized,
            'vector_db_size': len(self.vector_store) if self.vector_store else 0,
            'embedding_model': EMBEDDING_MODEL,
            'embedding_dimension': self.embedding_generator.dimension if self.embedding_generator else 0,
            'llm_model': LLM_MODEL,
            'ollama_url': OLLAMA_BASE_URL,
            'retrieval_config': {
                'top_k': TOP_K_CHUNKS,
                'relevance_threshold': RELEVANCE_THRESHOLD,
                'chunk_size': CHUNK_SIZE,
                'chunk_overlap': CHUNK_OVERLAP
            }
        }

    def _log(self, message: str, error: bool = False, warning: bool = False) -> None:
        """Internal logging"""
        if self.verbose:
            if error:
                logger.error(message)
                print(f"❌ {message}")
            elif warning:
                logger.warning(message)
                print(f"⚠️  {message}")
            else:
                logger.info(message)
                print(f"ℹ️  {message}")


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_mode(chatbot: CustomerSupportChatbot):
    """Run chatbot in interactive CLI mode"""
    print("\n" + "=" * 70)
    print("CUSTOMER SUPPORT CHATBOT")
    print("=" * 70)
    print("Type 'quit' or 'exit' to close")
    print("Type 'stats' to see statistics")
    print("Type 'reload' to rebuild knowledge base")
    print("=" * 70 + "\n")
    
    while True:
        try:
            question = input("You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = chatbot.get_stats()
                print(json.dumps(stats, indent=2))
                continue
            
            if question.lower() == 'reload':
                chatbot.reload_knowledge_base()
                continue
            
            # Get response
            response = chatbot.ask(question, return_sources=True)
            
            print(f"\nBot: {response['answer']}")
            print(f"Confidence: {response.get('confidence', 'unknown')}")
            
            if response.get('sources'):
                print("\nSources:")
                for source in response['sources']:
                    print(f"  - {source['source']} "
                          f"(chunk {source['chunk_id']}, "
                          f"similarity: {source['similarity']:.2f})")
            print()
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Create chatbot
    chatbot = CustomerSupportChatbot(verbose=True)
    
    # Initialize
    if not chatbot.initialize():
        print("Failed to initialize chatbot. Exiting.")
        sys.exit(1)
    
    # Run interactive mode
    interactive_mode(chatbot)
