"""
Embeddings Module
=================
Purpose: Convert text chunks into vector representations

This module handles:
1. Loading a pre-trained embedding model
2. Converting text to vectors (embeddings)
3. Each vector captures the semantic meaning of text

Why this matters for RAG:
- RAG searches by meaning, not keywords
- Vectors allow similarity searches
- "Payment term" and "Payment condition" get similar vectors
- FAISS will use these vectors to find relevant chunks
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    A class to generate embeddings (vector representations) of text.
    
    Think of this like: A translator that converts words into numbers 
    that represent their meaning.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            
        Why "all-MiniLM-L6-v2"?
        - Fast (good for production)
        - Accurate (good semantic understanding)
        - Small (fits on most computers)
        - Free (no API costs like OpenAI)
        
        First run will download the model (~50MB).
        Subsequent runs will use the cached model.
        """
        
        print(f"📥 Loading embedding model: {model_name}")
        print("   (First time takes 1-2 minutes, then uses cache)")
        
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded! Each embedding is {self.embedding_dim} dimensions")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of text chunks into embeddings.
        
        Args:
            texts (List[str]): List of text chunks
            
        Returns:
            np.ndarray: Matrix of embeddings (shape: len(texts) x embedding_dim)
            
        Example:
            >>> gen = EmbeddingGenerator()
            >>> chunks = ["Invoice number: 123", "Payment term: TT"]
            >>> embeddings = gen.generate_embeddings(chunks)
            >>> print(embeddings.shape)
            (2, 384)  # 2 chunks, 384 dimensions each
        """
        
        if not texts:
            raise ValueError("❌ No texts provided for embedding!")
        
        print(f"\n🔄 Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,  # Shows progress
            convert_to_numpy=True     # Returns numpy array (FAISS likes this)
        )
        
        print(f"✅ Embeddings generated!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   (Rows: chunks, Columns: dimensions)")
        
        return embeddings
    
    def get_embedding_for_query(self, query: str) -> np.ndarray:
        """
        Convert a single query string into an embedding.
        
        Args:
            query (str): The user's question
            
        Returns:
            np.ndarray: Single embedding vector
            
        Example:
            >>> gen = EmbeddingGenerator()
            >>> query = "What is the invoice number?"
            >>> query_embedding = gen.get_embedding_for_query(query)
            >>> print(query_embedding.shape)
            (384,)  # Single vector with 384 dimensions
        """
        
        if not query or not query.strip():
            raise ValueError("❌ Query is empty!")
        
        # Encode single query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        return query_embedding


# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Create embeddings generator
        gen = EmbeddingGenerator()
        
        # Example text chunks
        sample_chunks = [
            "Invoice Number: GTM-243054",
            "Payment Term: TT (Telegraphic Transfer)",
            "Shipper Line: Hapag Lloyd",
            "Shipment Term: FOB KARACHI PAKISTAN"
        ]
        
        # Generate embeddings
        embeddings = gen.generate_embeddings(sample_chunks)
        
        print("\n" + "="*50)
        print("EMBEDDING INFORMATION:")
        print("="*50)
        print(f"Number of chunks: {embeddings.shape[0]}")
        print(f"Dimensions per embedding: {embeddings.shape[1]}")
        print(f"\nFirst embedding (first 10 values):")
        print(embeddings[0][:10])
        
        # Test query embedding
        query = "What is the payment term?"
        query_emb = gen.get_embedding_for_query(query)
        print(f"\nQuery embedding shape: {query_emb.shape}")
        
    except Exception as e:
        print(f"Error: {e}")