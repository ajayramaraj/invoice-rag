"""
Retriever Module
================
Purpose: Retrieve relevant chunks from the vector store

This module handles:
1. Taking user questions
2. Converting them to embeddings
3. Searching FAISS for similar chunks
4. Returning top relevant chunks

Why this matters for RAG:
- User question → Embedding
- Embedding → FAISS search
- FAISS returns similar chunks
- Chunks become context for LLM
- Better context = Better answers
"""

from typing import List, Tuple
import numpy as np
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore


class Retriever:
    """
    A class to retrieve relevant chunks from the vector store.
    
    Think of this like: A smart search engine that finds 
    relevant documents based on semantic similarity.
    """
    
    def __init__(
        self, 
        embedding_generator: EmbeddingGenerator,
        vector_store: FAISSVectorStore,
        top_k: int = 3
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_generator (EmbeddingGenerator): Tool to embed queries
            vector_store (FAISSVectorStore): Indexed vector store
            top_k (int): Number of chunks to retrieve (default: 3)
        """
        
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.top_k = top_k
        
        print(f"🔍 Retriever initialized (returning top {top_k} chunks)")
    
    def retrieve(self, query: str) -> Tuple[List[str], List[float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query (str): User's question
            
        Returns:
            Tuple[List[str], List[float]]:
                - List of top_k relevant chunks
                - List of similarity distances
                
        Example:
            >>> retriever = Retriever(embedder, store, top_k=3)
            >>> query = "What is the invoice number?"
            >>> chunks, distances = retriever.retrieve(query)
            >>> print(chunks[0])
            "Invoice Number: GTM-243054"
        """
        
        if not query or not query.strip():
            raise ValueError("❌ Query is empty!")
        
        print(f"\n🔎 Retrieving chunks for: '{query}'")
        
        # Step 1: Convert query to embedding
        query_embedding = self.embedding_generator.get_embedding_for_query(query)
        
        # Step 2: Search FAISS for similar chunks
        retrieved_chunks, distances = self.vector_store.search(
            query_embedding, 
            k=self.top_k
        )
        
        # Step 3: Log results
        print(f"✅ Retrieved {len(retrieved_chunks)} chunks:")
        for i, (chunk, distance) in enumerate(zip(retrieved_chunks, distances), 1):
            # Show first 100 chars of chunk
            chunk_preview = chunk[:100].replace('\n', ' ')
            print(f"\n   {i}. (distance: {distance:.4f})")
            print(f"      {chunk_preview}...")
        
        return retrieved_chunks, distances
    
    def retrieve_with_scores(self, query: str) -> List[dict]:
        """
        Retrieve chunks and return as dictionaries with metadata.
        
        Args:
            query (str): User's question
            
        Returns:
            List[dict]: List of dicts with chunk, distance, and rank
            
        Example:
            >>> results = retriever.retrieve_with_scores(query)
            >>> for result in results:
            ...     print(result['chunk'])
            ...     print(f"Relevance: {result['relevance']}")
        """
        
        chunks, distances = self.retrieve(query)
        
        # Convert to dictionaries with more info
        results = []
        for rank, (chunk, distance) in enumerate(zip(chunks, distances), 1):
            # Convert distance to relevance score (0-1)
            # Lower distance = higher relevance
            relevance = 1 / (1 + distance)  # Sigmoid-like conversion
            
            results.append({
                'rank': rank,
                'chunk': chunk,
                'distance': float(distance),
                'relevance': float(relevance)  # 0 to 1 scale
            })
        
        return results
    
    def get_context(self, query: str) -> str:
        """
        Get retrieved chunks as a formatted context string.
        
        Args:
            query (str): User's question
            
        Returns:
            str: Formatted context for LLM
            
        This is what gets passed to the LLM prompt.
        """
        
        chunks, _ = self.retrieve(query)
        
        # Format chunks as context
        context = "RETRIEVED CONTEXT:\n"
        context += "=" * 50 + "\n"
        for i, chunk in enumerate(chunks, 1):
            context += f"\n[Chunk {i}]\n{chunk}\n"
            context += "-" * 50 + "\n"
        
        return context


# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Note: This is just a demo. In real usage:
        # 1. EmbeddingGenerator will embed actual PDF chunks
        # 2. FAISSVectorStore will have real embeddings
        # 3. Retriever will find actual relevant chunks
        
        print("🚀 Retriever Module Demo")
        print("(This is a simplified example)")
        print("\nNote: Actual usage requires:")
        print("  1. Real PDF text chunks")
        print("  2. Embedded chunks in FAISS")
        print("  3. User query")
        
    except Exception as e:
        print(f"Error: {e}")