import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os
import ast

class ChromaDBService:
    """Service for managing ChromaDB vector store."""

    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        print(f"[INFO] Initializing ChromaDB at: {self.persist_directory}")
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def get_or_create_collection(self, collection_name: str = "vehicle_manuals"):
        """Creates or gets a ChromaDB collection."""
        # Using default embedding function (all-MiniLM-L6-v2) or we can pass our own embeddings
        # Since we are generating embeddings using sentence-transformers externally, 
        # we will pass the embeddings directly when adding documents.
        return self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, chunks: list[dict], collection_name: str = "vehicle_manuals"):
        """
        Adds text chunks and their embeddings to the collection.
        Args:
            chunks: List of dictionaries containing 'sentence_chunk', 'embedding', and metadata.
            collection_name: Name of the collection.
        """
        collection = self.get_or_create_collection(collection_name)
        
        print(f"[INFO] Adding {len(chunks)} documents to collection: {collection_name}")
        
        ids = [f"id_{i}" for i in range(len(chunks))]
        documents = [item["sentence_chunk"] for item in chunks]
        embeddings = [item["embedding"] for item in chunks]
        
        # Prepare metadata (filter out non-primitive types if necessary)
        metadatas = []
        for item in chunks:
            # Create a copy and remove large fields if needed
            meta = {
                "page_number": item.get("page_number"),
                "chunk_char_count": item.get("chunk_char_count"),
                "chunk_word_count": item.get("chunk_word_count"),
                "chunk_token_count": item.get("chunk_token_count"),
                "pdf_file": item.get("pdf_file", "unknown")
            }
            metadatas.append(meta)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print("[INFO] Documents added successfully.")

    def query(self, query_embeddings: list, n_results: int = 5, collection_name: str = "vehicle_manuals"):
        """
        Queries the collection using embeddings.
        Args:
            query_embeddings: List of embedding vectors.
            n_results: Number of results to return.
        Returns:
            Query results.
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )

    def reset_collection(self, collection_name: str = "vehicle_manuals"):
        """
        Deletes and recreates the collection to remove all data.
        """
        try:
            self.client.delete_collection(name=collection_name)
            print(f"[INFO] Collection '{collection_name}' deleted.")
        except ValueError:
            print(f"[WARN] Collection '{collection_name}' does not exist.")
        
        self.get_or_create_collection(collection_name)
        print(f"[INFO] Collection '{collection_name}' recreated.")

if __name__ == "__main__":
    import sys
    import numpy as np
    
    # Path setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "text_chunks_and_embeddings.csv")
    chroma_path = os.path.join(data_dir, "chroma_store")

    try:
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}. Please run embeddings.py first.")
            sys.exit(1)

        print(f"[INFO] Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Convert string representation of embedding back to list
        # Using ast.literal_eval is safer than eval()
        print("[INFO] parsing embeddings...")
        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        
        chunks = df.to_dict(orient="records")

        # Initialize ChromaDB
        chroma_service = ChromaDBService(persist_directory=chroma_path)
        
        # Add documents
        chroma_service.add_documents(chunks)
        
        # Verify with a fake query (using the first embedding as a query)
        print("[INFO] Verifying with a test query...")
        test_embedding = chunks[0]["embedding"]
        results = chroma_service.query(query_embeddings=[test_embedding], n_results=2)
        
        print("\n[RESULT] Top 2 matches:")
        print(results["documents"][0])
        
    except Exception as e:
        print(f"Error: {e}")
