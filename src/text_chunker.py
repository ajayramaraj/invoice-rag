"""
Text Chunker Module
===================

Purpose: Split large text into smaller, manageable chunks

This module handles:
1. Taking long extracted text
2. Splitting into smaller pieces (chunks)
3. Keeping related content together
4. Handling overlap between chunks

Why this matters for RAG:
- RAG systems work better with smaller pieces
- Each chunk becomes a separate document for embedding
- Overlap helps context not get cut off in the middle
"""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    """
    A class to split large text into smaller chunks.

    Think of this like:
    A machine that cuts a long document into
    smaller pieces that are still related to each other.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size (int):
                Maximum characters per chunk.

            chunk_overlap (int):
                Number of overlapping characters between chunks.
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
            length_function=len,
        )

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Args:
            text (str):
                Extracted PDF text.

        Returns:
            List[str]:
                List of chunks.
        """

        if not text or not text.strip():
            raise ValueError(
                "❌ Text is empty! Nothing to chunk."
            )

        chunks = self.splitter.split_text(text)

        print(f"✅ Text split into {len(chunks)} chunks")
        print(f"📦 Chunk Size: {self.chunk_size}")
        print(f"🔄 Overlap: {self.chunk_overlap}")

        return chunks


# ==========================================================
# TESTING SECTION
# ==========================================================

if __name__ == "__main__":

    print("\n🚀 Testing Text Chunker...\n")

    chunker = TextChunker(
        chunk_size=1000,
        chunk_overlap=200
    )

    sample_text = """
    Invoice Number: GTM-243054
    Date: 2024-01-15

    Buyer: ABC Corporation
    Seller: XYZ Traders

    Payment Terms: TT (Telegraphic Transfer)
    Due Date: 2024-02-15

    Shipper Line: Hapag Lloyd
    Shipment Terms: FOB KARACHI PAKISTAN

    Item Details:

    Item 1:
    Cotton Fabric
    Quantity: 100 meters
    Amount: $5000

    Item 2:
    Polyester Thread
    Quantity: 50 kg
    Amount: $1000

    Item 3:
    Dyes and Chemicals
    Quantity: 200 liters
    Amount: $3000

    Financial Summary:

    Subtotal: $9000
    Tax (5%): $450
    Total Amount: $9450

    Bank Details:

    Bank Name:
    First National Bank

    Account Number:
    123456789

    SWIFT Code:
    FNBAPK

    Notes:
    Please transfer funds within 30 days.

    Contact:
    invoice@xyztraders.com
    """

    try:

        chunks = chunker.split_text(sample_text)

        print("\n" + "=" * 70)
        print("CHUNK DETAILS")
        print("=" * 70)

        for i, chunk in enumerate(chunks, start=1):

            print(f"\n📄 Chunk {i}")
            print("-" * 70)
            print(f"Length: {len(chunk)} characters")
            print("-" * 70)
            print(chunk)

        print("\n" + "=" * 70)
        print(f"✅ Total Chunks Created: {len(chunks)}")
        print("=" * 70)

    except Exception as e:

        print("\n❌ ERROR OCCURRED")
        print(str(e))