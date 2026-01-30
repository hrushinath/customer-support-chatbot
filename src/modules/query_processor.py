"""
Query Processor Module
Handles query embedding and retrieval of relevant chunks from vector DB
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Processes user queries and retrieves relevant context from vector DB.
    
    Pipeline:
    1. Embed user query (same model as chunks)
    2. Search vector DB for similar chunks
    3. Filter by relevance threshold
    4. Rank results
    5. Return top-K chunks as context
    """

    def __init__(
        self,
        embedding_generator,  # EmbeddingGenerator instance
        vector_store,  # FAISSVectorStore instance
        top_k: int = 5,
        relevance_threshold: float = 0.5,
        min_chunk_length: int = 50
    ):
        """
        Initialize query processor
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator
            vector_store: Instance of FAISSVectorStore or ChromaVectorStore
            top_k: Number of top results to return
            relevance_threshold: Minimum similarity score (0-1)
            min_chunk_length: Minimum chunk length to consider
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.min_chunk_length = min_chunk_length
        
        logger.info(
            f"QueryProcessor initialized: "
            f"top_k={top_k}, threshold={relevance_threshold}"
        )

    def process_query(self, query: str) -> List[Tuple[Dict, float]]:
        """
        Process a user query and retrieve relevant chunks
        
        Args:
            query: User question/query text
            
        Returns:
            List of (metadata_dict, similarity_score) tuples,
            sorted by similarity descending
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Embed the query
        query_embedding = self.embedding_generator.embed_text(query)
        logger.debug(f"Query embedding shape: {query_embedding.shape}")
        
        # Step 2: Search vector DB
        raw_results = self.vector_store.search(query_embedding, k=self.top_k * 2)
        
        if not raw_results:
            logger.info("No results found in vector DB")
            return []
        
        logger.debug(f"Retrieved {len(raw_results)} raw results")
        
        # Step 3: Filter and process results
        filtered_results = []
        
        for chunk_id, similarity, metadata in raw_results:
            # Log all results for debugging
            chunk_preview = metadata.get('text', '')[:100]
            logger.info(f"Chunk {chunk_id}: similarity={similarity:.4f}, preview='{chunk_preview}...'")
            
            # Check relevance threshold
            if similarity < self.relevance_threshold:
                logger.debug(
                    f"Skipped chunk (low relevance): "
                    f"sim={similarity:.4f} < {self.relevance_threshold}"
                )
                continue
            
            # Check minimum chunk length
            chunk_length = metadata.get('length', len(metadata.get('text', '')))
            if chunk_length < self.min_chunk_length:
                logger.debug(
                    f"Skipped chunk (too short): "
                    f"len={chunk_length} < {self.min_chunk_length}"
                )
                continue
            
            # Add metadata about retrieval
            metadata['similarity_score'] = similarity
            metadata['chunk_id'] = chunk_id
            
            filtered_results.append((metadata, similarity))
        
        # Step 4: Sort by similarity
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # Step 5: Return top-K
        final_results = filtered_results[:self.top_k]
        
        logger.info(
            f"Retrieved {len(final_results)} relevant chunks "
            f"(avg similarity: {np.mean([s for _, s in final_results]):.4f})"
        )
        
        return final_results

    def get_context_text(
        self,
        query: str,
        separator: str = "\n\n---\n\n"
    ) -> Tuple[str, List[Dict]]:
        """
        Get combined context text from retrieved chunks
        
        Args:
            query: User query
            separator: Text to separate chunks
            
        Returns:
            Tuple of (combined_context_text, source_info_list)
        """
        results = self.process_query(query)
        
        if not results:
            return "", []
        
        # Combine chunk texts
        chunk_texts = [meta['text'] for meta, _ in results]
        context = separator.join(chunk_texts)
        
        # Prepare source information
        sources = [
            {
                'source': meta.get('source', 'unknown'),
                'chunk_id': meta.get('chunk_id', -1),
                'similarity': meta.get('similarity_score', 0),
                'length': meta.get('length', 0)
            }
            for meta, _ in results
        ]
        
        logger.debug(f"Combined context length: {len(context)} chars")
        
        return context, sources

    def get_answer_context(
        self,
        query: str,
        include_metadata: bool = True
    ) -> Dict:
        """
        Get comprehensive context for answer generation
        
        Args:
            query: User query
            include_metadata: Include detailed metadata
            
        Returns:
            Dict with:
            - 'context': Combined text context
            - 'sources': List of source info
            - 'confidence': Confidence based on average similarity
            - 'chunks_count': Number of chunks retrieved
            - 'combined_length': Total context length
        """
        results = self.process_query(query)
        
        if not results:
            return {
                'context': '',
                'sources': [],
                'confidence': 'low',
                'chunks_count': 0,
                'combined_length': 0,
                'no_results': True
            }
        
        # Combine contexts
        context_text = '\n\n---\n\n'.join([meta['text'] for meta, _ in results])
        
        # Calculate confidence
        similarities = [sim for _, sim in results]
        avg_similarity = np.mean(similarities)
        
        # FAISS cosine similarity scores are typically 0.3-0.9 range
        if avg_similarity >= 0.45:
            confidence = 'high'
        elif avg_similarity >= 0.30:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Prepare sources
        sources = [
            {
                'source': meta.get('source', 'unknown'),
                'chunk_id': meta.get('chunk_id', -1),
                'similarity': round(meta.get('similarity_score', 0), 4),
                'type': meta.get('doc_type', 'unknown')
            }
            for meta, _ in results
        ]
        
        return {
            'context': context_text,
            'sources': sources if include_metadata else [],
            'confidence': confidence,
            'chunks_count': len(results),
            'combined_length': len(context_text),
            'avg_similarity': round(avg_similarity, 4),
            'no_results': False
        }

    def set_parameters(
        self,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = None
    ) -> None:
        """
        Dynamically adjust retrieval parameters
        
        Args:
            top_k: New k value
            relevance_threshold: New threshold
        """
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Updated top_k to {top_k}")
        
        if relevance_threshold is not None:
            if not (0 <= relevance_threshold <= 1):
                raise ValueError("Relevance threshold must be between 0 and 1")
            self.relevance_threshold = relevance_threshold
            logger.info(f"Updated relevance_threshold to {relevance_threshold}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("QUERY PROCESSOR EXAMPLE")
    print("=" * 70)
    print("\nNote: This requires a loaded vector store.")
    print("See the main app.py for complete usage example.\n")
