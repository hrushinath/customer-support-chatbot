"""
Embeddings Module
Generates vector embeddings for text chunks using Sentence-Transformers
"""

import logging
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Sentence-Transformers for semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates semantic embeddings for text using Sentence-Transformers.
    
    Embeddings capture semantic meaning, allowing similarity search.
    Two semantically similar texts will have similar embedding vectors.
    
    Example:
        "What is your return policy?" → [0.45, -0.23, 0.89, ..., 0.34]
        "Can I return items?" → [0.48, -0.21, 0.87, ..., 0.35]
        (These would be similar vectors)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Path = None
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model ID
                Options:
                - "sentence-transformers/all-MiniLM-L6-v2" (384 dims, fast) ✓ Recommended
                - "sentence-transformers/all-mpnet-base-v2" (768 dims, slower)
                - "sentence-transformers/BGE-base-en-v1.5" (768 dims, excellent)
            device: "cpu", "cuda", or "mps" (Apple Silicon)
            cache_folder: Where to cache models
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers required. Install with:\n"
                "pip install sentence-transformers"
            )
        
        logger.info(f"Loading embedding model: {model_name}")
        
        self.model_name = model_name
        self.device = device
        
        # Load the model (will download on first use)
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            f"Embedding model loaded. "
            f"Dimensions: {self.embedding_dim}, Device: {device}"
        )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array (shape: (embedding_dim,))
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dim)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Process this many texts at once (higher = faster but more RAM)
            
        Returns:
            List of embedding vectors
        """
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings

    def embed_chunks(
        self,
        chunks: List,  # List of Chunk objects
        batch_size: int = 32
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        Generate embeddings for a list of chunks
        
        Args:
            chunks: List of Chunk objects from text_chunker
            batch_size: Texts to process at once
            
        Returns:
            List of (embedding_vector, chunk_metadata) tuples
        """
        # Extract chunk texts
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(chunk_texts, batch_size=batch_size)
        
        # Pair embeddings with chunk metadata
        result = []
        for embedding, chunk in zip(embeddings, chunks):
            metadata = {
                'text': chunk.text,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'start_idx': chunk.start_idx,
                'end_idx': chunk.end_idx,
                'doc_type': chunk.doc_type,
                'length': len(chunk.text)
            }
            result.append((embedding, metadata))
        
        logger.info(f"Successfully embedded {len(result)} chunks")
        return result

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate similarity between query and many corpus embeddings
        (Optimized using numpy operations)
        
        Args:
            query_embedding: Single query embedding
            corpus_embeddings: Array of corpus embeddings (n_docs x embedding_dim)
            
        Returns:
            Array of similarity scores
        """
        # Normalize
        query_norm = np.linalg.norm(query_embedding)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
        
        if query_norm == 0:
            return np.zeros(len(corpus_embeddings))
        
        # Cosine similarity
        similarities = np.dot(corpus_embeddings, query_embedding) / (corpus_norms.flatten() * query_norm)
        return similarities

    @property
    def dimension(self) -> int:
        """Return embedding dimension"""
        return self.embedding_dim


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize generator
    generator = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )
    
    # Example texts
    texts = [
        "What is your return policy?",
        "Can I send items back?",
        "How long does shipping take?",
        "The product is very good",
        "I love this purchase"
    ]
    
    print("\n=== Embedding Demonstration ===\n")
    
    # Embed the texts
    embeddings = generator.embed_texts(texts)
    
    print(f"Generated embeddings for {len(texts)} texts")
    print(f"Embedding dimension: {generator.embedding_dim}")
    print(f"First embedding shape: {embeddings[0].shape}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}\n")
    
    # Calculate similarities between first text and others
    print("Similarity to 'What is your return policy?':")
    for i, text in enumerate(texts):
        if i > 0:
            sim = generator.similarity(embeddings[0], embeddings[i])
            print(f"  {text[:40]:<40} → {sim:.4f}")
    
    # Demonstrate batch similarity
    print("\nBatch similarity calculation:")
    query_emb = embeddings[0]
    corpus_embs = np.array(embeddings[1:])
    sims = generator.batch_similarity(query_emb, corpus_embs)
    print(f"Similarities: {sims}")
