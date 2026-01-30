"""
Vector Database Module
Stores embeddings in FAISS (Facebook AI Similarity Search)
Enables fast retrieval of relevant chunks
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Vector database using FAISS (Facebook AI Similarity Search).
    
    Stores embeddings and enables fast similarity search:
    - O(1) storage, O(n) search time for exact search
    - Can scale to billions of vectors
    - CPU-optimized, no GPU required
    - Fully local, no cloud dependency
    """

    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embedding vectors (e.g., 384 for all-MiniLM)
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu required. Install with:\n"
                "pip install faiss-cpu\n"
                "or for GPU: pip install faiss-gpu"
            )
        
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        self.metadata = []  # Store chunk metadata
        
        logger.info(f"Initialized FAISS index with dimension {embedding_dim}")

    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict]
    ) -> None:
        """
        Add embeddings and metadata to the index
        
        Args:
            embeddings: List of embedding vectors (numpy arrays)
            metadata_list: List of metadata dicts for each embedding
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype(np.float32)
        
        # Normalize for cosine similarity (FAISS uses L2 distance)
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.metadata.extend(metadata_list)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Prepare query
        query_array = np.array([query_embedding]).astype(np.float32)
        faiss.normalize_L2(query_array)
        
        # Search
        distances, indices = self.index.search(query_array, k=min(k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # FAISS returns L2 distance, convert to similarity
            # For normalized vectors: similarity = 1 - (distance/2)
            similarity = 1 - (dist / 2)
            
            if idx < len(self.metadata):
                metadata = self.metadata[int(idx)]
                results.append((int(idx), float(similarity), metadata))
        
        return results

    def save(self, index_path: Path, metadata_path: Path) -> None:
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata JSON
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        # Ensure directories exist
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

    @classmethod
    def load(cls, index_path: Path, metadata_path: Path) -> 'FAISSVectorStore':
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            
        Returns:
            Loaded FAISSVectorStore instance
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load index
        index = faiss.read_index(str(index_path))
        embedding_dim = index.d
        
        # Create store
        store = cls(embedding_dim)
        store.index = index
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            store.metadata = json.load(f)
        
        logger.info(
            f"Loaded FAISS index from {index_path} "
            f"({index.ntotal} embeddings)"
        )
        
        return store

    def __len__(self) -> int:
        """Return number of stored embeddings"""
        return self.index.ntotal

    def clear(self) -> None:
        """Clear all data from index"""
        self.index.reset()
        self.metadata = []
        logger.info("Cleared FAISS index")


class ChromaVectorStore:
    """
    Vector database using ChromaDB.
    More feature-rich alternative to FAISS.
    Requires: pip install chromadb
    """

    def __init__(self, persist_directory: Path = None):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_directory: Directory to persist data
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb required. Install with:\n"
                "pip install chromadb"
            )
        
        self.persist_directory = Path(persist_directory) if persist_directory else Path("./chroma_db")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        
        # Create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB at {self.persist_directory}")

    def add_embeddings(
        self,
        embeddings: List[np.ndarray],
        metadata_list: List[Dict]
    ) -> None:
        """Add embeddings and metadata to ChromaDB"""
        ids = [f"chunk_{i}" for i in range(len(embeddings))]
        
        self.collection.add(
            ids=ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=metadata_list,
            documents=[m.get('text', '') for m in metadata_list]
        )
        
        logger.info(f"Added {len(embeddings)} embeddings to ChromaDB")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search for similar embeddings"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        output = []
        for i, (doc_id, distance, metadata) in enumerate(zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            output.append((i, similarity, metadata))
        
        return output


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a small vector store
    store = FAISSVectorStore(embedding_dim=384)
    
    # Add some dummy embeddings
    embeddings = [
        np.random.randn(384).astype(np.float32) for _ in range(10)
    ]
    
    metadata = [
        {"text": f"Sample text {i}", "source": "test.txt", "chunk_id": i}
        for i in range(10)
    ]
    
    store.add_embeddings(embeddings, metadata)
    
    # Search with a query
    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, k=3)
    
    print("\nSearch results:")
    for chunk_id, sim, meta in results:
        print(f"  Chunk {meta['chunk_id']}: similarity={sim:.4f}")
    
    # Save and load
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test.index"
        meta_path = Path(tmpdir) / "test_metadata.json"
        
        store.save(index_path, meta_path)
        print(f"\nSaved to {tmpdir}")
        
        loaded_store = FAISSVectorStore.load(index_path, meta_path)
        print(f"Loaded store with {len(loaded_store)} embeddings")
