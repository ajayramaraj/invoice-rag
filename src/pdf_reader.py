"""
PDF Reader Module
=================
Purpose: Read PDF files and extract text

This module handles:
1. Loading PDF files from disk
2. Extracting all text from all pages
3. Returning clean, readable text

Why this matters for RAG:
- RAG needs TEXT input (not PDFs)
- We extract text here, then split it into chunks
- This is Step 1 of the RAG pipeline
"""

from pathlib import Path
from pypdf import PdfReader
from typing import Optional


class PDFReader:
    """
    A class to read and extract text from PDF files.
    
    Think of this like: A machine that reads PDFs and gives you pure text.
    """
    
    def __init__(self):
        """Initialize the PDF reader."""
        pass
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Read a PDF file and extract all text from it.
        
        Args:
            pdf_path (str): Full path to the PDF file
            
        Returns:
            str: All text extracted from the PDF
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF reading fails
        """
        
        # Convert string path to Path object (modern Python way)
        pdf_file = Path(pdf_path)
        
        # Check if file exists
        if not pdf_file.exists():
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")
        
        # Check if it's actually a PDF
        if pdf_file.suffix.lower() != ".pdf":
            raise ValueError(f"❌ File is not a PDF: {pdf_path}")
        
        try:
            # Open and read the PDF
            pdf_reader = PdfReader(pdf_file)
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"📄 PDF loaded: {num_pages} pages found")
            
            # Extract text from ALL pages
            extracted_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Remove extra whitespace and clean up
            extracted_text = extracted_text.strip()
            
            print(f"✅ Text extracted successfully! Total length: {len(extracted_text)} characters")
            
            return extracted_text
            
        except Exception as e:
            raise Exception(f"❌ Error reading PDF: {str(e)}")


# Example usage (for testing)
if __name__ == "__main__":
    # This code runs only if you run this file directly
    reader = PDFReader()
    
    # Test with a PDF (we'll create one later)
    try:
        text = reader.extract_text_from_pdf("data/invoice.pdf")
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(text[:500])  # Print first 500 characters
    except Exception as e:
        print(f"Error: {e}")