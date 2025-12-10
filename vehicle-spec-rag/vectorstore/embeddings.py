import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

class EmbeddingService:
    """Service for creating and managing text embeddings."""

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = None):
        if device is None:
            # Auto-detect device: CUDA -> MPS (Mac) -> CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"[INFO] Initializing EmbeddingService with model: {model_name} on device: {self.device}")
        self.model = SentenceTransformer(model_name_or_path=model_name, device=self.device)

    def generate_embeddings(self, chunks: list[dict] | list[str], batch_size: int = 32) -> list:
        """
        Generates embeddings for a list of text chunks or strings.
        Args:
            chunks: List of dictionaries containing 'sentence_chunk' OR list of strings.
            batch_size: Batch size for encoding.
        Returns:
            List of embeddings (numpy arrays or list).
        """
        if isinstance(chunks[0], dict):
            text_chunks = [item["sentence_chunk"] for item in chunks]
        else:
            text_chunks = chunks
        
        print("[INFO] Generating embeddings...")
        
        # Encode all chunks at once (batched library side)
        embeddings = self.model.encode(
            text_chunks, 
            batch_size=batch_size, 
            convert_to_tensor=False, # Return numpy arrays/list for easier compatibility
            show_progress_bar=True
        )

        # Convert to list if needed (depending on return type) but usually numpy array
        # Return the embeddings directly. The caller is responsible for assigning them.
        return embeddings

    def save_embeddings(self, chunks: list[dict], file_path: str):
        """Saves chunks and embeddings to a CSV file."""
        df = pd.DataFrame(chunks)
        print(f"[INFO] Saving {len(df)} embeddings to {file_path}")
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    import os
    import sys
    
    # Path setup
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pdf_processing.extract_text import PDFTextExtractor
    from pdf_processing.chunker import TextChunker

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "sample-service-manual 1.pdf")
    output_path = os.path.join(base_dir, "data", "text_chunks_and_embeddings.csv")

    try:
        # 1. Extract
        extractor = PDFTextExtractor()
        pages_and_text = extractor.extract(pdf_path)

        # 2. Chunk
        chunker = TextChunker(min_token_length=100)
        chunks = chunker.chunk(pages_and_text)

        # 3. Embed
        embedder = EmbeddingService() # Auto-detects device
        embeddings = embedder.generate_embeddings(chunks)
        
        # Assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        # 4. Save
        embedder.save_embeddings(chunks, output_path)

        print("[INFO] Pipeline complete.")
        
    except Exception as e:
        print(f"Error: {e}")
