"""
Text Chunking Module
Intelligently splits documents into overlapping chunks for better RAG retrieval
"""

import logging
from typing import List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    source: str  # Original document filename
    chunk_id: int  # Sequential ID within document
    start_idx: int  # Character index in original document
    end_idx: int  # Character index in original document
    doc_type: str  # Document type (pdf, txt, json, etc.)


class TextChunker:
    """
    Intelligently chunks text for RAG systems.
    
    Chunking strategy:
    - Respects paragraph/sentence boundaries
    - Creates overlapping chunks to preserve context
    - Maintains metadata for source attribution
    - Filters out very small chunks
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        min_chunk_length: int = 50
    ):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Target size of each chunk in characters (~4 chars = 1 token)
            chunk_overlap: Number of characters to overlap between chunks
            separator: String to split on (try: "\n\n", "\n", ".")
            min_chunk_length: Minimum characters for a chunk to be kept
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.min_chunk_length = min_chunk_length
        
        logger.info(
            f"TextChunker initialized: "
            f"size={chunk_size}, overlap={chunk_overlap}, sep='{repr(separator)}'"
        )

    def chunk_text(
        self,
        text: str,
        source: str,
        doc_type: str = "unknown"
    ) -> List[Chunk]:
        """
        Split text into chunks
        
        Args:
            text: The document text to chunk
            source: Original document filename/source
            doc_type: Type of document (txt, pdf, json, etc.)
            
        Returns:
            List of Chunk objects
        """
        if not text or len(text.strip()) == 0:
            logger.warning(f"Empty document: {source}")
            return []
        
        # First, split by primary separator (usually paragraph)
        segments = text.split(self.separator)
        
        # Merge segments into chunks of target size
        chunks = self._merge_segments(segments, source, doc_type)
        
        # Filter out small chunks
        chunks = [c for c in chunks if len(c.text) >= self.min_chunk_length]
        
        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks

    def _merge_segments(
        self,
        segments: List[str],
        source: str,
        doc_type: str
    ) -> List[Chunk]:
        """
        Merge segments into chunks of target size with overlap
        
        Args:
            segments: List of text segments
            source: Document source name
            doc_type: Document type
            
        Returns:
            List of merged chunks
        """
        chunks = []
        chunk_id = 0
        current_chunk = ""
        start_idx = 0
        current_idx = 0
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            # Add separator before segment (except first)
            if current_chunk:
                segment_with_sep = self.separator + segment
            else:
                segment_with_sep = segment
            
            # If adding this segment would exceed chunk_size, save current chunk
            if current_chunk and len(current_chunk) + len(segment_with_sep) > self.chunk_size:
                # Create chunk from current_chunk
                if current_chunk.strip():
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        source=source,
                        chunk_id=chunk_id,
                        start_idx=start_idx,
                        end_idx=current_idx,
                        doc_type=doc_type
                    )
                    chunks.append(chunk)
                    chunk_id += 1
                
                # Start new chunk with overlap
                # Go back by overlap amount
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + segment_with_sep
                start_idx = current_idx - len(overlap_text)
            else:
                current_chunk += segment_with_sep
            
            current_idx += len(segment_with_sep)
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunk = Chunk(
                text=current_chunk.strip(),
                source=source,
                chunk_id=chunk_id,
                start_idx=start_idx,
                end_idx=current_idx,
                doc_type=doc_type
            )
            chunks.append(chunk)
        
        return chunks

    def chunk_documents(
        self,
        documents: List,  # List of Document objects from document_loader
        verbose: bool = True
    ) -> List[Chunk]:
        """
        Chunk multiple documents at once
        
        Args:
            documents: List of Document objects
            verbose: Print progress
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            if verbose:
                logger.info(f"Chunking: {doc.filename}")
            
            chunks = self.chunk_text(
                text=doc.content,
                source=doc.filename,
                doc_type=doc.doc_type
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


# ============================================================================
# ADVANCED CHUNKING STRATEGIES
# ============================================================================

class SmartChunker(TextChunker):
    """
    Enhanced chunking that preserves semantic boundaries
    (Requires more computation but better quality chunks)
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Use sentence boundary as separator
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=".",  # Split on sentence
            min_chunk_length=50
        )

    def chunk_text(self, text: str, source: str, doc_type: str = "unknown") -> List[Chunk]:
        """
        Smart chunking that respects sentence boundaries
        """
        # Clean up text
        text = text.replace('\n\n', ' [PARAGRAPH] ')
        text = text.replace('\n', ' ')
        text = text.replace(' [PARAGRAPH] ', '\n\n')
        
        return super().chunk_text(text, source, doc_type)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example text
    sample_text = """
    Customer Support FAQ
    
    Q: What is your return policy?
    A: We accept returns within 30 days of purchase. Items must be in original condition 
    with tags attached. You'll need a receipt or proof of purchase.
    
    Q: How long does shipping take?
    A: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.
    Shipping is free for orders over $50.
    
    Q: What payment methods do you accept?
    A: We accept all major credit cards, PayPal, Apple Pay, and Google Pay.
    We also offer installment plans through Afterpay and Klarna.
    """
    
    # Create chunker
    chunker = TextChunker(
        chunk_size=300,
        chunk_overlap=100,
        separator="\n",
        min_chunk_length=30
    )
    
    # Chunk the text
    chunks = chunker.chunk_text(sample_text, "faq.txt", "txt")
    
    print(f"\nCreated {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"Chunk #{chunk.chunk_id}:")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Text: {chunk.text[:80]}...")
        print()
