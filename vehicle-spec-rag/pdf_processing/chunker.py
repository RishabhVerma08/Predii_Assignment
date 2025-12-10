import re
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English

class TextChunker:
    """Service for splitting text into sentence chunks."""

    def __init__(self, sentence_chunk_size: int = 10, min_token_length: int = 30):
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.sentence_chunk_size = sentence_chunk_size
        self.min_token_length = min_token_length

    @staticmethod
    def _split_list(input_list: list, slice_size: int) -> list:
        """Splits a list into sublists of size slice_size."""
        return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

    def chunk(self, pages_and_text: list[dict]) -> list[dict]:
        """
        Chunks the extracted text into groups of sentences.
        Args:
            pages_and_text: List of dicts containing 'text' and other metadata per page.
        Returns:
            List of dicts containing chunked text and metadata.
        """
        print("[INFO] Starting text chunking...")
        
        # 1. Split text into sentences
        for item in tqdm(pages_and_text, desc="Splitting into sentences"):
            item["sentences"] = list(self.nlp(item["text"]).sents)
            item["sentences"] = [str(sentence) for sentence in item["sentences"]]
            item["page_sentence_count_spacy"] = len(item["sentences"])

        # 2. Chunk sentences
        for item in tqdm(pages_and_text, desc="Grouping sentences"):
            item["sentence_chunks"] = self._split_list(input_list=item["sentences"],
                                                       slice_size=self.sentence_chunk_size)
            item["num_chunks"] = len(item["sentence_chunks"])

        # 3. Create chunk dicts
        pages_and_chunks = []
        for item in tqdm(pages_and_text, desc="Creating chunk objects"):
            for sentence_chunk in item["sentence_chunks"]:
                chunk_dict = {}
                chunk_dict["page_number"] = item["page_number"]
                
                # Join sentences into a paragaph
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

                chunk_dict["sentence_chunk"] = joined_sentence_chunk

                chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
                chunk_dict["chunk_word_count"] = len([word for word in joined_sentence_chunk.split(" ")])
                chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

                pages_and_chunks.append(chunk_dict)

        # 4. Filter short chunks
        df = pd.DataFrame(pages_and_chunks)
        print(f"[INFO] Initial chunks: {len(df)}")
        
        pages_and_chunks_over_min_token_len = df[df["chunk_token_count"] > self.min_token_length].to_dict(orient="records")
        print(f"[INFO] Chunks after filtering (token > {self.min_token_length}): {len(pages_and_chunks_over_min_token_len)}")

        return pages_and_chunks_over_min_token_len

if __name__ == "__main__":
    import os
    import sys
    
    # Add project root to path to import other modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pdf_processing.extract_text import PDFTextExtractor

    # Setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "sample-service-manual 1.pdf")

    try:
        # Extract
        extractor = PDFTextExtractor()
        pages_and_text = extractor.extract(pdf_path)

        # Chunk
        chunker = TextChunker(min_token_length=100) # Using higher threshold for demo/verification
        chunks = chunker.chunk(pages_and_text)

        # Show results
        print(f"\nExample Chunk:\n{chunks[0] if chunks else 'No chunks found'}")
        
    except Exception as e:
        print(f"Error: {e}")
