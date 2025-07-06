import re
import numpy as np
import os
import openai
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import json
from dotenv import load_dotenv

load_dotenv()

    
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
    embedding: Optional[np.ndarray] = None

class TextChunker:
    """Handles intelligent text chunking with context preservation"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, source_url: str) -> List[TextChunk]:
        """
        Chunk text into overlapping segments that preserve context
        
        Args:
            text: The text to chunk
            source_url: URL of the source page
            
        Returns:
            List of TextChunk objects
        """
        # Clean the text
        text = self._clean_text(text)
        
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
                    end_pos=current_start + len(current_chunk)
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
                end_pos=current_start + len(current_chunk)
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

def TextEmbedder(query):
    """Handles text embedding using Azure OpenAI"""


    """Get embedding from Azure OpenAI using full 1536 dimensions."""
    response = client.embeddings.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input=query,
        encoding_format="float" 
    )

    print(f'Azure open ai deployment name: {AZURE_OPENAI_DEPLOYMENT}')

    embedding = np.array(response.data[0].embedding, dtype=np.float32)  # Ensure FAISS-compatible float32 format

    #take out this statement later
    assert embedding.shape[0] == 1536, f"Unexpected embedding dimension: {embedding.shape[0]}"

    return embedding.reshape(1, -1)

def get_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Get embeddings for multiple texts at once"""
    embeddings = []
    for text in texts:
        embedding = TextEmbedder(text)
        embeddings.append(embedding.flatten())  # Flatten to 1D for stacking
    return np.array(embeddings)

class SimilaritySearcher:
    """Handles cosine similarity search between queries and chunks"""
    
    def __init__(self, chunks: List[TextChunk]):
        self.chunks = chunks
        self._build_index()
    
    def _build_index(self):
        """Build the embedding index for all chunks"""
        chunk_texts = [chunk.content for chunk in self.chunks]
        embeddings = get_embeddings_batch(chunk_texts)
        
        # Store embeddings in chunks
        for i, chunk in enumerate(self.chunks):
            chunk.embedding = embeddings[i]
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """
        Search for most similar chunks to the query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity
        """
        # Embed the query
        query_embedding = TextEmbedder(query).flatten()
        
        # Calculate similarities
        similarities = []
        json_doc=[]
        for chunk in self.chunks:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                chunk.embedding.reshape(1, -1)
            )[0][0]
            json_doc.append({
        "source_url": chunk.source_url,
        "content": chunk.content,
        "chunk_id":chunk.chunk_id,
        "start_pos":chunk.start_pos,
        "end_pos":chunk.end_pos,
        "similarity": float(similarity)
    })
            similarities.append((chunk, similarity))
        
        with open("chunks.json", "w", encoding="utf-8") as f:
            json.dump(json_doc, f, indent=2, ensure_ascii=False)
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> Dict[str, List[Tuple[TextChunk, float]]]:
        """Search for multiple queries at once"""
        results = {}
        for query in queries:
            results[query] = self.search(query, top_k)
        return results

class TextProcessingPipeline:
    """Complete pipeline for processing scraped text"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunker = TextChunker(chunk_size, overlap)
        self.chunks = []
        self.searcher = None
    
    def process_scraped_data(self, scraped_data: Dict[str, str]) -> List[TextChunk]:
        """
        Process scraped data into chunks and build search index
        
        Args:
            scraped_data: Dictionary of {url: text_content}
            
        Returns:
            List of all text chunks
        """
        all_chunks = []
        
        for url, text in scraped_data.items():
            chunks = self.chunker.chunk_text(text, url)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        self.searcher = SimilaritySearcher(self.chunks)
        
        return all_chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[TextChunk, float]]:
        """Search for relevant chunks"""
        if not self.searcher:
            raise ValueError("Must process data first using process_scraped_data()")
        return self.searcher.search(query, top_k)
    
    def get_chunk_stats(self) -> Dict:
        """Get statistics about the chunks"""
        if not self.chunks:
            return {}
        
        chunk_lengths = [len(chunk.content) for chunk in self.chunks]
        url_counts = {}
        for chunk in self.chunks:
            url_counts[chunk.source_url] = url_counts.get(chunk.source_url, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'avg_chunk_length': np.mean(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'chunks_per_url': url_counts
        }

def chunk_to_dict(chunk: TextChunk) -> Dict:
    """Convert TextChunk to dictionary for JSON serialization"""
    chunk_dict = asdict(chunk)
    # Remove embedding as it's not JSON serializable
    chunk_dict.pop('embedding', None)
    return chunk_dict

# Example usage
if __name__ == "__main__":
    # Example scraped data
    with open("scraped_content.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    char_limit = 4000
    scraped_data = {data["url"]: data["content"][:char_limit]}
    
    # Initialize and run pipeline
    pipeline = TextProcessingPipeline(chunk_size=500, overlap=50)
    chunks = pipeline.process_scraped_data(scraped_data)
    
    # Search for relevant chunks
    query = "what is the latest on fine-tuning techniques for machine learning"
    results = pipeline.search(query, top_k=2)
    
    # 2. Write top K chunks metadata to relevant_chunks.json
    relevant_chunks_metadata = []
    for chunk, score in results:
        chunk_dict = chunk_to_dict(chunk)
        chunk_dict['similarity_score'] = float(score)  # Add similarity score
        relevant_chunks_metadata.append(chunk_dict)
    
    with open("relevant_chunks.json", "w", encoding="utf-8") as f:
        json.dump(relevant_chunks_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to chunks.json")
    print(f"Saved {len(results)} relevant chunks to relevant_chunks.json")
    print(f"Query: {query}")
    print(f"Top {len(results)} relevant chunks saved with similarity scores")