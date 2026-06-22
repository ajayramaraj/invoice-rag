"""
Vector Store Module (FAISS)
===========================
Purpose: Store and search embeddings using FAISS

This module handles:
1. Creating a FAISS index from embeddings
2. Storing embeddings in an indexed, searchable format
3. Saving/loading the index to disk
4. Searching for similar vectors (similarity search)

Why this matters for RAG:
- Raw embeddings are slow to search (O(n) complexity)
- FAISS indexes them for O(log n) fast search
- Can search through millions of vectors in milliseconds
"""

import os
from typing import List, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """
    A class to store and search embeddings using FAISS.
    
    Think of this like: A super-fast library card catalog 
    for finding similar vectors.
    """
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the FAISS vector store.
        
        Args:
            embedding_dim (int): Dimension of embeddings (default: 384 from our model)
        """
        
        self.embedding_dim = embedding_dim
        self.index = None
        self.texts = []  # Store original texts (for retrieval)
        self.is_trained = False
        
        print(f"🗄️  Vector Store initialized (dimension: {embedding_dim})")
    
    def create_index(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Create and train the FAISS index from embeddings.
        
        Args:
            embeddings (np.ndarray): Matrix of embeddings (shape: n_texts x embedding_dim)
            texts (List[str]): List of original text chunks
            
        Example:
            >>> embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
            >>> texts = ["chunk1", "chunk2"]
            >>> store.create_index(embeddings, texts)
        """
        
        if len(embeddings) != len(texts):
            raise ValueError("❌ Number of embeddings must match number of texts!")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"❌ Embedding dimension mismatch! "
                f"Expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )
        
        print(f"\n📊 Creating FAISS index...")
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Number of texts: {len(texts)}")
        
        # Convert embeddings to float32 (FAISS requirement)
        embeddings = np.array(embeddings).astype('float32')
        
        # Create a simple flat index (fast for smaller datasets)
        # For larger datasets (millions), use IVF or HNSW indexes
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store original texts
        self.texts = texts
        self.is_trained = True
        
        print(f"✅ FAISS index created and trained!")
        print(f"   Total vectors in index: {self.index.ntotal}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """
        Search for the k most similar embeddings.
        
        Args:
            query_embedding (np.ndarray): Single query embedding (shape: embedding_dim)
            k (int): Number of results to return (default: 3)
            
        Returns:
            Tuple[List[str], List[float]]: 
                - List of k most similar texts
                - List of distances (lower = more similar)
                
        Example:
            >>> query_emb = np.array([0.1, 0.2, 0.3])
            >>> texts, distances = store.search(query_emb, k=3)
            >>> print(texts)
            ["Invoice Number: 123", "Invoice Date: 2024-01-15", ...]
        """
        
        if not self.is_trained:
            raise ValueError("❌ Index not trained! Call create_index() first.")
        
        if len(query_embedding.shape) == 1:
            # If single vector, reshape to (1, dim)
            query_embedding = query_embedding.reshape(1, -1)
        
        # Convert to float32
        query_embedding = query_embedding.astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Unpack results (they're in nested arrays)
        distances = distances[0].tolist()
        indices = indices[0].tolist()
        
        # Get the texts
        retrieved_texts = [self.texts[i] for i in indices]
        
        return retrieved_texts, distances
    
    def save_index(self, save_dir: str = "vectorstore") -> None:
        """
        Save the FAISS index and texts to disk.
        
        Args:
            save_dir (str): Directory to save files
        """
        
        if not self.is_trained:
            raise ValueError("❌ Index not trained! Nothing to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        faiss.write_index(self.index, index_path)
        
        # Save texts (using simple format)
        texts_path = os.path.join(save_dir, "texts.txt")
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in self.texts:
                # Escape newlines so each text is one line
                escaped_text = text.replace('\n', '\\n')
                f.write(escaped_text + '\n')
        
        print(f"✅ Index saved!")
        print(f"   Index: {index_path}")
        print(f"   Texts: {texts_path}")
    
    def load_index(self, save_dir: str = "vectorstore") -> None:
        """
        Load the FAISS index and texts from disk.
        
        Args:
            save_dir (str): Directory where files are saved
        """
        
        # Load FAISS index
        index_path = os.path.join(save_dir, "faiss_index.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"❌ Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load texts
        texts_path = os.path.join(save_dir, "texts.txt")
        if not os.path.exists(texts_path):
            raise FileNotFoundError(f"❌ Texts file not found: {texts_path}")
        
        with open(texts_path, 'r', encoding='utf-8') as f:
            self.texts = [line.rstrip('\n').replace('\\n', '\n') for line in f]
        
        self.is_trained = True
        
        print(f"✅ Index loaded!")
        print(f"   Total vectors: {self.index.ntotal}")
        print(f"   Total texts: {len(self.texts)}")


# Example usage (for testing)
if __name__ == "__main__":
    try:
        # Create vector store
        store = FAISSVectorStore(embedding_dim=384)
        
        # Example embeddings (random for testing)
        sample_embeddings = np.random.randn(4, 384).astype('float32')
        sample_texts = [
            "Invoice Number: GTM-243054",
            "Payment Term: TT (Telegraphic Transfer)",
            "Shipper Line: Hapag Lloyd",
            "Shipment Term: FOB KARACHI PAKISTAN"
        ]
        
        # Create index
        store.create_index(sample_embeddings, sample_texts)
        
        # Example query
        query_embedding = np.random.randn(384).astype('float32')
        retrieved_texts, distances = store.search(query_embedding, k=2)
        
        print("\n" + "="*50)
        print("SEARCH RESULTS:")
        print("="*50)
        for i, (text, distance) in enumerate(zip(retrieved_texts, distances), 1):
            print(f"\n{i}. {text}")
            print(f"   Distance: {distance:.4f}")
        
        # Save index
        store.save_index()
        
        # Load index (simulate new session)
        print("\n" + "="*50)
        print("LOADING INDEX FROM DISK:")
        print("="*50)
        store2 = FAISSVectorStore(embedding_dim=384)
        store2.load_index()
        
    except Exception as e:
        print(f"Error: {e}")