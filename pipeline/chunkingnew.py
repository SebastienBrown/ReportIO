import re
import numpy as np
import os
import openai
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from contextlib import contextmanager

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Azure OpenAI Configuration from .env
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = "2023-05-15"

client = openai.AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION
)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    source_url: str
    chunk_id: int
    start_pos: int
    end_pos: int
    document_id: str  # New field to identify which document this chunk belongs to
    embedding: Optional[np.ndarray] = None

@dataclass
class ProcessingStats:
    """Statistics about the processing pipeline"""
    total_documents: int
    total_chunks: int
    avg_chunks_per_document: float
    processing_time: float
    embedding_time: float
    failed_embeddings: int = 0

class TextChunker:
    """Handles intelligent text chunking with context preservation"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_url: str, document_id: str) -> List[TextChunk]:
        """
        Chunk text into overlapping segments that preserve context
        
        Args:
            text: The text to chunk
            source_url: URL of the source page
            document_id: Unique identifier for this document
            
        Returns:
            List of TextChunk objects
        """
        # Clean the text
        text = self._clean_text(text)
        
        if not text.strip():
            return []
        
        # Split into sentences for better context preservation
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    source_url=source_url,
                    chunk_id=chunk_id,
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    document_id=document_id
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                current_chunk = overlap_text + " " + sentence
                current_start = current_start + len(current_chunk) - len(overlap_text) - 1
                chunk_id += 1
            else:
                if not current_chunk:
                    current_start = text.find(sentence)
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                source_url=source_url,
                chunk_id=chunk_id,
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                document_id=document_id
            )
            chunks.append(chunk)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
        return text.strip()
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last overlap_size characters, preferring sentence boundaries"""
        if len(text) <= overlap_size:
            return text
        
        # Try to find a sentence boundary within the overlap region
        overlap_start = len(text) - overlap_size
        overlap_text = text[overlap_start:]
        
        # Look for sentence ending
        sentence_ends = ['.', '!', '?']
        for i, char in enumerate(overlap_text):
            if char in sentence_ends and i > overlap_size // 2:
                return overlap_text[i+1:].strip()
        
        return overlap_text

class EmbeddingManager:
    """Handles text embedding with rate limiting and batch processing"""
    
    def __init__(self, max_workers: int = 5, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self.failed_embeddings = 0
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Azure OpenAI"""
        try:
            response = client.embeddings.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                input=text,
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to get embedding for text: {e}")
            with self._lock:
                self.failed_embeddings += 1
            return None
    
    def get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts with parallel processing"""
        embeddings = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_text = {executor.submit(self.get_embedding, text): text for text in batch}
                
                for future in as_completed(future_to_text):
                    embedding = future.result()
                    embeddings.append(embedding)
        
        return embeddings
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Add embeddings to chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.get_embeddings_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks

class BackendTextProcessor:
    """Main processor for handling multiple documents in a backend environment"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, max_workers: int = 5):
        self.chunker = TextChunker(chunk_size, overlap)
        self.embedding_manager = EmbeddingManager(max_workers=max_workers)
        self.all_chunks: List[TextChunk] = []
        self.document_registry: Dict[str, Dict] = {}
        self.processing_stats = None
        self._lock = threading.Lock()
    
    def add_document(self, text: str, source_url: str, document_id: str = None) -> List[TextChunk]:
        """
        Add a single document to the processing pipeline
        
        Args:
            text: Document content
            source_url: Source URL
            document_id: Unique identifier (auto-generated if not provided)
            
        Returns:
            List of chunks created from this document
        """
        if document_id is None:
            document_id = f"doc_{len(self.document_registry)}"
        
        # Chunk the text
        chunks = self.chunker.chunk_text(text, source_url, document_id)
        
        # Register the document
        with self._lock:
            self.document_registry[document_id] = {
                'source_url': source_url,
                'chunk_count': len(chunks),
                'text_length': len(text),
                'processed': False
            }
        
        logger.info(f"Added document {document_id} with {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[Dict[str, str]]) -> List[TextChunk]:
        """
        Process multiple documents at once
        
        Args:
            documents: List of dicts with 'text', 'source_url', and optionally 'document_id'
            
        Returns:
            List of all chunks from all documents
        """
        import time
        start_time = time.time()
        
        all_chunks = []
        
        # Process each document
        for i, doc in enumerate(documents):
            text = doc.get('content', '')
            source_url = doc.get('url', '')
            document_id = doc.get('title', f"doc_{i}")
            print("source_url is ",source_url)
            
            if not text:
                logger.warning(f"Empty text for document {document_id}")
                continue
            
            chunks = self.add_document(text, source_url, document_id)
            all_chunks.extend(chunks)
        
        processing_time = time.time() - start_time
        
        # Generate embeddings for all chunks
        embedding_start = time.time()
        all_chunks = self.embedding_manager.embed_chunks(all_chunks)
        embedding_time = time.time() - embedding_start
        
        # Store chunks
        with self._lock:
            self.all_chunks = all_chunks
            
            # Update registry
            for doc_id in self.document_registry:
                self.document_registry[doc_id]['processed'] = True
        
        # Create stats
        self.processing_stats = ProcessingStats(
            total_documents=len(documents),
            total_chunks=len(all_chunks),
            avg_chunks_per_document=len(all_chunks) / len(documents) if documents else 0,
            processing_time=processing_time,
            embedding_time=embedding_time,
            failed_embeddings=self.embedding_manager.failed_embeddings
        )
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks in {processing_time:.2f}s")
        return all_chunks
    
    def search(self, query: str, top_k: int = 5, filter_by_document: List[str] = None) -> List[Tuple[TextChunk, float]]:
        """
        Search for relevant chunks across all processed documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_by_document: Optional list of document IDs to filter by
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        if not self.all_chunks:
            raise ValueError("No documents processed yet. Call process_documents() first.")
        
        # Filter chunks if specified
        chunks_to_search = self.all_chunks
        if filter_by_document:
            chunks_to_search = [chunk for chunk in self.all_chunks if chunk.document_id in filter_by_document]
        
        # Get query embedding
        query_embedding = self.embedding_manager.get_embedding(query)
        if query_embedding is None:
            raise ValueError("Failed to get embedding for query")
        
        # Calculate similarities
        similarities = []
        for chunk in chunks_to_search:
            if chunk.embedding is not None:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    chunk.embedding.reshape(1, -1)
                )[0][0]
                similarities.append((chunk, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the processing pipeline"""
        stats = {
            'document_count': len(self.document_registry),
            'total_chunks': len(self.all_chunks),
            'processed_documents': sum(1 for doc in self.document_registry.values() if doc['processed']),
            'document_registry': self.document_registry
        }
        
        if self.processing_stats:
            stats.update(asdict(self.processing_stats))
        
        if self.all_chunks:
            chunk_lengths = [len(chunk.content) for chunk in self.all_chunks]
            stats.update({
                'avg_chunk_length': np.mean(chunk_lengths),
                'min_chunk_length': min(chunk_lengths),
                'max_chunk_length': max(chunk_lengths),
                'chunks_by_document': {doc_id: sum(1 for chunk in self.all_chunks if chunk.document_id == doc_id) 
                                     for doc_id in self.document_registry}
            })
        
        return stats
    
    def save_results(self, query: str, results: List[Tuple[TextChunk, float]], 
                    chunks_file: str = "all_chunks.json", 
                    results_file: str = "best_chunks.json"):
        """Save chunks and search results to files"""
        
        # Save all chunks
        chunks_data = []
        for chunk in self.all_chunks:
            chunk_dict = self._chunk_to_dict(chunk)
            chunks_data.append(chunk_dict)
        
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        # Save search results
        results_data = {
            'query': query,
            'results': []
        }
        
        for chunk, score in results:
            result_dict = self._chunk_to_dict(chunk)
            result_dict['similarity_score'] = float(score)
            results_data['results'].append(result_dict)
        
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.all_chunks)} chunks to {chunks_file}")
        logger.info(f"Saved {len(results)} search results to {results_file}")
    
    def _chunk_to_dict(self, chunk: TextChunk) -> Dict:
        """Convert TextChunk to dictionary for JSON serialization"""
        chunk_dict = asdict(chunk)
        # Remove embedding as it's not JSON serializable
        chunk_dict.pop('embedding', None)
        return chunk_dict
    
    def clear(self):
        """Clear all processed data"""
        with self._lock:
            self.all_chunks.clear()
            self.document_registry.clear()
            self.processing_stats = None
        logger.info("Cleared all processed data")

# Backend Service Class
class TextProcessingService:
    """Service class for easy backend integration"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, max_workers: int = 5):
        self.processor = BackendTextProcessor(chunk_size, overlap, max_workers)
    
    def process_batch(self, scraped_data: List[Dict]) -> Dict:
        """
        Process a batch of scraped data
        
        Args:
            scraped_data: List of dicts with 'text', 'source_url', and optionally 'document_id'
            
        Returns:
            Processing results and statistics
        """
        try:
            chunks = self.processor.process_documents(scraped_data)
            stats = self.processor.get_stats()
            
            return {
                'success': True,
                'chunk_count': len(chunks),
                'stats': stats,
                'message': f"Successfully processed {len(scraped_data)} documents into {len(chunks)} chunks"
            }
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return {
                'success': False,
                'error': str(e),
                'chunk_count': 0
            }
    
    def search_and_save(self, query: str, top_k: int = 5, output_dir: str = ".") -> Dict:
        """
        Search and save results to files
        
        Args:
            query: Search query
            top_k: Number of results to return
            output_dir: Directory to save files
            
        Returns:
            Search results and metadata
        """
        try:
            results = self.processor.search(query, top_k)
            
            # Save results
            chunks_file = os.path.join(output_dir, "all_chunks.json")
            results_file = os.path.join(output_dir, "best_chunks.json")
            
            self.processor.save_results(query, results, chunks_file, results_file)
            
            return {
                'success': True,
                'query': query,
                'results_count': len(results),
                'files_saved': [chunks_file, results_file],
                'top_results': [
                    {
                        'document_id': chunk.document_id,
                        'source_url': chunk.source_url,
                        'similarity_score': float(score),
                        'content_preview': chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                    }
                    for chunk, score in results[:3]  # Preview of top 3
                ]
            }
        except Exception as e:
            logger.error(f"Error in search_and_save: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_service_stats(self) -> Dict:
        """Get service statistics"""
        return self.processor.get_stats()
    
    def reset_service(self):
        """Reset the service for a new batch"""
        self.processor.clear()

# Example usage for backend integration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize service
    service = TextProcessingService(chunk_size=500, overlap=50, max_workers=3)
    
    # Example: Process multiple documents
    documents = [
        {
            'text': "This is the first document about AI and machine learning...",
            'source_url': "https://example.com/doc1",
            'document_id': "doc1"
        },
        {
            'text': "This is the second document about natural language processing...",
            'source_url': "https://example.com/doc2",
            'document_id': "doc2"
        }
    ]
    
    # Process documents
    processing_result = service.process_batch(documents)
    print("Processing result:", processing_result)
    
    # Search and save
    if processing_result['success']:
        search_result = service.search_and_save(
            query="",
            top_k=5,
            output_dir="."
        )
        print("Search result:", search_result)
    
    # Get stats
    stats = service.get_service_stats()
    print("Service stats:", stats)