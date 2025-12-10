import os
import sys

# Ensure we can import modules from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectorstore.chroma_db import ChromaDBService
from vectorstore.embeddings import EmbeddingService

class Retriever:
    """Service for retrieving documents relevant to a query."""

    def __init__(self, chroma_service: ChromaDBService, embedding_service: EmbeddingService):
        self.chroma_service = chroma_service
        self.embedding_service = embedding_service

    def retrieve(self, query: str, k: int = 5, collection_name: str = "vehicle_manuals") -> list[str]:
        """
        Retrieves top k documents relevant to the query string.
        Args:
            query: The user query string.
            k: Number of documents to retrieve.
            collection_name: Target collection.
        Returns:
            List of document texts.
        """
        print(f"[INFO] Retrieving top {k} documents for query: '{query}'")
        
        # 1. Generate embedding for the query
        # user model.encode returns a numpy array or tensor, we need to make sure it's a list for Chroma
        query_embedding = self.embedding_service.model.encode(query, convert_to_tensor=False).tolist()

        # 2. Retrieve based on embedding
        return self.retrieve_by_embedding(query_embedding, k, collection_name)

    def retrieve_by_embedding(self, query_embedding: list, k: int = 5, collection_name: str = "vehicle_manuals") -> list[str]:
        """
        Retrieves top k documents based on a pre-computed embedding.
        Args:
            query_embedding: The query embedding vector (list).
            k: Number of documents to retrieve.
            collection_name: Target collection.
        Returns:
            List of document texts.
        """
        # Wrap in list because query expects a list of embeddings
        results = self.chroma_service.query(query_embeddings=[query_embedding], n_results=k, collection_name=collection_name)
        
        # Extract documents from results
        # results['documents'] is a list of lists (one list per query)
        if results and results['documents']:
            return results['documents'][0]
        return []

if __name__ == "__main__":
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_path = os.path.join(base_dir, "data", "chroma_store")
    
    try:
        # Initialize services
        embedder = EmbeddingService()
        chroma = ChromaDBService(persist_directory=chroma_path)
        
        # Initialize Retriever
        retriever = Retriever(chroma_service=chroma, embedding_service=embedder)
        
        # Test Query
        query = "How do I inspect the suspension system?"
        results = retriever.retrieve(query, k=3)
        
        print(f"\n[QUERY] {query}")
        print(f"[RESULTS FOUND] {len(results)}")
        for i, doc in enumerate(results):
            print(f"-- Result {i+1} --")
            print(doc)
            print()
            
    except Exception as e:
        print(f"Error: {e}")
