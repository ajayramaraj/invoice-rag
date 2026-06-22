"""
RAG Pipeline Module
===================
Purpose: Orchestrate the entire RAG pipeline

This module handles:
1. Reading and chunking PDFs
2. Creating and storing embeddings
3. Retrieving relevant chunks
4. Generating answers with LLM

Pipeline flow:
PDF → Extract → Chunk → Embed → Store → Retrieve → Generate → Answer

Why this matters:
- Ties all modules together
- Single interface for the entire system
- Easy to use from Streamlit
- Professional architecture
"""

from typing import List, Tuple, Optional
import os
from pathlib import Path

from src.pdf_reader import PDFReader
from src.text_chunker import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.retriever import Retriever
from src.llm import GroqLLM


class RAGPipeline:
    """
    Complete RAG Pipeline - orchestrates all components.
    
    Think of this like: A conductor managing an entire orchestra,
    ensuring all instruments play in harmony.
    """
    
    def __init__(
        self,
        groq_api_key: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        vectorstore_dir: str = "vectorstore"
    ):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            groq_api_key (str): Groq API key
            chunk_size (int): Text chunk size
            chunk_overlap (int): Overlap between chunks
            top_k (int): Number of chunks to retrieve
            vectorstore_dir (str): Directory to save/load FAISS index
        """
        
        print("🚀 Initializing RAG Pipeline...")
        print("="*50)
        
        # Initialize components
        self.pdf_reader = PDFReader()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedding_generator.embedding_dim
        )
        self.llm = GroqLLM(api_key=groq_api_key)
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            top_k=top_k
        )
        
        self.vectorstore_dir = vectorstore_dir
        self.is_indexed = False
        
        print("="*50)
        print("✅ RAG Pipeline initialized!")
        print(f"   Chunk size: {chunk_size}")
        print(f"   Overlap: {chunk_overlap}")
        print(f"   Top K: {top_k}")
        print(f"   Vector store dir: {vectorstore_dir}")
    
    def index_pdf(self, pdf_path: str) -> None:
        """
        Index a PDF file through the entire pipeline.
        
        Steps:
        1. Read PDF
        2. Chunk text
        3. Generate embeddings
        4. Store in FAISS
        5. Save index to disk
        
        Args:
            pdf_path (str): Path to PDF file
        """
        
        print("\n" + "="*50)
        print("📄 STEP 1: Reading PDF...")
        print("="*50)
        
        # Step 1: Read PDF
        text = self.pdf_reader.extract_text_from_pdf(pdf_path)
        
        print("\n" + "="*50)
        print("✂️  STEP 2: Chunking Text...")
        print("="*50)
        
        # Step 2: Chunk text
        chunks = self.text_chunker.split_text(text)
        
        print("\n" + "="*50)
        print("🔢 STEP 3: Generating Embeddings...")
        print("="*50)
        
        # Step 3: Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(chunks)
        
        print("\n" + "="*50)
        print("🗄️  STEP 4: Creating FAISS Index...")
        print("="*50)
        
        # Step 4: Create FAISS index
        self.vector_store.create_index(embeddings, chunks)
        
        print("\n" + "="*50)
        print("💾 STEP 5: Saving Index...")
        print("="*50)
        
        # Step 5: Save index
        self.vector_store.save_index(self.vectorstore_dir)
        
        self.is_indexed = True
        
        print("\n" + "="*50)
        print("✅ PDF Indexed Successfully!")
        print("="*50)
    
    def load_index(self) -> None:
        """
        Load a previously saved FAISS index.
        
        Useful when:
        - User already indexed a PDF
        - Want to ask new questions without re-indexing
        - Faster startup
        """
        
        print("\n📂 Loading FAISS index from disk...")
        self.vector_store.load_index(self.vectorstore_dir)
        self.is_indexed = True
        print("✅ Index loaded!")
    
    def answer_question(self, query: str) -> str:
        """
        Answer a question using the RAG pipeline.
        
        Steps:
        1. Retrieve relevant chunks
        2. Format as context
        3. Generate answer with LLM
        
        Args:
            query (str): User's question
            
        Returns:
            str: LLM-generated answer
        """
        
        if not self.is_indexed:
            raise ValueError(
                "❌ Index not loaded! Call index_pdf() or load_index() first."
            )
        
        print("\n" + "="*50)
        print("🔍 STEP 6: Retrieving Relevant Chunks...")
        print("="*50)
        
        # Step 6: Retrieve chunks
        chunks, distances = self.retriever.retrieve(query)
        
        # Format context
        context = self.retriever.get_context(query)
        
        print("\n" + "="*50)
        print("🤖 STEP 7: Generating Answer with LLM...")
        print("="*50)
        
        # Step 7: Generate answer
        answer = self.llm.generate_answer(query, context)
        
        return answer
    
    def answer_question_with_context(self, query: str) -> dict:
        """
        Answer question and return answer with retrieved context.
        
        Returns:
            dict: Contains answer, retrieved chunks, and metadata
        """
        
        if not self.is_indexed:
            raise ValueError("❌ Index not loaded!")
        
        # Retrieve chunks
        chunks, distances = self.retriever.retrieve(query)
        context = self.retriever.get_context(query)
        
        # Generate answer
        answer = self.llm.generate_answer(query, context)
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': chunks,
            'distances': distances,
            'num_chunks': len(chunks)
        }


# Example usage (for testing)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
    else:
        try:
            print("🚀 RAG Pipeline Demo")
            print("="*50)
            
            # Create pipeline
            pipeline = RAGPipeline(groq_api_key=api_key)
            
            # Path to PDF
            pdf_path = "data/invoice.pdf"
            
            if os.path.exists(pdf_path):
                # Index PDF
                pipeline.index_pdf(pdf_path)
                
                # Ask questions
                questions = [
                    "What is the invoice number?",
                    "What is the payment term?",
                    "What is the shipper line?",
                    "What is the shipment term?"
                ]
                
                print("\n" + "="*50)
                print("ANSWERING QUESTIONS")
                print("="*50)
                
                for question in questions:
                    answer = pipeline.answer_question(question)
                    print(f"\nQ: {question}")
                    print(f"A: {answer}")
            else:
                print(f"❌ PDF not found: {pdf_path}")
                print("Please upload a PDF to data/ folder first")
            
        except Exception as e:
            print(f"❌ Error: {e}")