import requests
import json
import numpy as np
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import time
import logging
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass
from dotenv import load_dotenv
import os


load_dotenv()

@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    similarity_score: float = 0.0

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


def get_openai_embedding(text):
    """Get embedding from Azure OpenAI using full 1536 dimensions."""
    response = client.embeddings.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        input=text,
        encoding_format="float" 
    )

    print(f'Azure open ai deployment name: {AZURE_OPENAI_DEPLOYMENT}')


    embedding = np.array(response.data[0].embedding, dtype=np.float32)  # Ensure FAISS-compatible float32 format

    #take out this statement later
    assert embedding.shape[0] == 1536, f"Unexpected embedding dimension: {embedding.shape[0]}"

    return embedding.reshape(1, -1)  



def calculate_similarity(query: str, json_path):
        """
        Calculate similarity between query and search result snippets from JSON input using Azure OpenAI embeddings.
        
        Args:
            query: The search query string
            json_results: List of dictionaries with 'title', 'link', and 'snippet' keys
            
        Returns:
            List of SearchResult objects sorted by similarity score
        """
        with open(json_path, "r", encoding="utf-8") as f:
            json_results = json.load(f)

        if not json_results:
            return []

        # Get query embedding
        query_embedding = get_openai_embedding(query)

        # Get embeddings for all snippets
        snippet_embeddings = []
        for item in json_results:
            snippet = item.get("snippet", "")
            snippet_embedding = get_openai_embedding(snippet)
            snippet_embeddings.append(snippet_embedding[0])  # Remove extra dimension if needed
            
        # Convert to numpy array
        snippet_embeddings = np.array(snippet_embeddings)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, snippet_embeddings)[0]
        
        # Update results with similarity scores
        for i, item in enumerate(json_results):
            item["similarity_score"] = similarities[i]

        # Sort results by similarity
        sorted_results = sorted(json_results, key=lambda x: x["similarity_score"], reverse=True)

        # Print top 5
        print("Calculated similarity scores for all results:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. {result.get('title', '')} (Score: {result['similarity_score']:.4f})")
            print(f"   URL: {result.get('link', '')}")
            print(f"   Snippet: {result.get('snippet', '')[:100]}...\n")

        return sorted_results

        
calculate_similarity("Latest AI research breakthroughs","search_results.json")

