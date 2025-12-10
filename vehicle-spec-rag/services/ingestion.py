
import os
import shutil
from typing import List

from pdf_processing.extract_text import PDFTextExtractor
from pdf_processing.chunker import TextChunker
from vectorstore.embeddings import EmbeddingService
from vectorstore.chroma_db import ChromaDBService

class IngestionService:
    """Orchestrates the data ingestion pipeline."""

    def __init__(self, chroma_service: ChromaDBService, embedding_service: EmbeddingService):
        self.chroma_service = chroma_service
        self.embedding_service = embedding_service
        self.pdf_extractor = PDFTextExtractor()
        self.chunker = TextChunker()

    def process_file(self, file_path: str, collection_name: str = "vehicle_manuals"):
        """
        Full pipeline:
        1. Extract Text
        2. Chunk Text
        3. Embed Chunks
        4. Reset DB & Store
        """
        print(f"[INFO] Starting ingestion for: {file_path}")
        
        # 1. Extract
        pdf_data = self.pdf_extractor.extract(file_path)
        print(f"[INFO] Extracted {len(pdf_data)} pages.")
        
        # 2. Chunk
        # TextChunker.chunk expects a list of dicts with 'text' and 'page_number'
        # which is exactly what PDFTextExtractor.extract returns.
        all_chunks = self.chunker.chunk(pdf_data)
        
        # Add pdf_file metadata
        for chunk in all_chunks:
            chunk["pdf_file"] = os.path.basename(file_path)
        
        print(f"[INFO] Created {len(all_chunks)} chunks.")
        
        
        # 3. Embed
        # Extract just texts for embedding
        texts = [c["sentence_chunk"] for c in all_chunks]
        embeddings = self.embedding_service.generate_embeddings(texts)
        
        # Add embeddings back to chunk dicts
        for i, chunk in enumerate(all_chunks):
            # Convert numpy to list for Chroma
            chunk["embedding"] = embeddings[i].tolist()

        # 4. Reset & Store
        self.chroma_service.reset_collection(collection_name)
        self.chroma_service.add_documents(all_chunks, collection_name)
        
        print("[INFO] Ingestion complete.")
        return len(all_chunks)
