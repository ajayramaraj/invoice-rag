"""
LLM Module (Groq Integration)
=============================
Purpose: Generate answers using Groq LLM with retrieved context

This module handles:
1. Connecting to Groq API
2. Creating prompts with context
3. Calling the LLM with context
4. Returning generated answers

Why this matters for RAG:
- Context chunks alone are not answers
- LLM understands and synthesizes context
- Groq is fast, cheap, and accurate
- We use llama-3.3-70b model (open source, powerful)
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


class GroqLLM:
    """
    A class to generate answers using Groq LLM.
    
    Think of this like: An interface to a smart AI that 
    reads context and generates intelligent answers.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1024
    ):
        """
        Initialize Groq LLM.
        
        Args:
            api_key (str): Groq API key from environment
            model (str): Model to use (default: llama-3.3-70b-versatile)
            temperature (float): Creativity level (0=precise, 1=creative)
            max_tokens (int): Max output length
            
        Why these settings?
        - temperature=0.3: We want accurate, deterministic answers (not creative)
        - max_tokens=1024: Enough for detailed answers, not too much
        - llama-3.3-70b: Powerful, open-source, fast on Groq
        """
        
        if not api_key:
            raise ValueError("❌ API key is empty! Set GROQ_API_KEY in .env")
        
        print(f"🤖 Initializing Groq LLM...")
        print(f"   Model: {model}")
        print(f"   Temperature: {temperature}")
        print(f"   Max tokens: {max_tokens}")
        
        # Initialize Groq ChatLLM (modern LangChain way)
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        print(f"✅ Groq LLM initialized!")
    
    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate an answer based on query and context.
        
        Args:
            query (str): User's question
            context (str): Retrieved chunks formatted as context
            system_prompt (str): Optional custom system instructions
            
        Returns:
            str: LLM-generated answer
            
        Example:
            >>> llm = GroqLLM(api_key="...")
            >>> answer = llm.generate_answer(
            ...     query="What is the invoice number?",
            ...     context="[Chunk 1] Invoice Number: GTM-243054..."
            ... )
            >>> print(answer)
            "The invoice number is GTM-243054"
        """
        
        if not query or not query.strip():
            raise ValueError("❌ Query is empty!")
        
        if not context or not context.strip():
            raise ValueError("❌ Context is empty!")
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """You are an expert invoice processing assistant.
Your task is to answer questions about invoices based on the provided context.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the answer is not in the context, say "Information not found in the invoice"
3. Be concise and accurate
4. Extract exact values (numbers, dates, names) without modification
5. If asked about something not in the invoice, do NOT make up information"""
        
        print(f"\n📝 Generating answer...")
        print(f"   Query: {query}")
        print(f"   Context length: {len(context)} chars")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:")
        ])
        
        # Format the prompt
        formatted_prompt = prompt.format(context=context, query=query)
        
        # Call LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Extract text from response
        answer = response.content.strip()
        
        print(f"✅ Answer generated!")
        print(f"   Length: {len(answer)} characters")
        
        return answer
    
    def generate_answer_with_reasoning(
        self,
        query: str,
        context: str
    ) -> dict:
        """
        Generate answer with reasoning steps.
        
        Args:
            query (str): User's question
            context (str): Retrieved chunks
            
        Returns:
            dict: Answer with metadata
        """
        
        answer = self.generate_answer(query, context)
        
        return {
            'query': query,
            'answer': answer,
            'context_length': len(context),
            'model': self.model,
            'temperature': self.temperature
        }


# Example usage (for testing)
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env file!")
        print("Please add: GROQ_API_KEY=your_key_here to .env")
    else:
        try:
            # Create LLM
            llm = GroqLLM(api_key=api_key)
            
            # Example context
            context = """
            [Chunk 1]
            Invoice Number: GTM-243054
            Invoice Date: 2024-01-15
            
            [Chunk 2]
            Payment Term: TT (Telegraphic Transfer)
            Due Date: 2024-02-15
            
            [Chunk 3]
            Shipper Line: Hapag Lloyd
            Shipment Term: FOB KARACHI PAKISTAN
            """
            
            # Test query
            query = "What is the invoice number?"
            
            # Generate answer
            answer = llm.generate_answer(query, context)
            
            print("\n" + "="*50)
            print("ANSWER:")
            print("="*50)
            print(answer)
            
        except Exception as e:
            print(f"❌ Error: {e}")