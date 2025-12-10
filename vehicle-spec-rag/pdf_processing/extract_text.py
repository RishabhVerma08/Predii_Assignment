import os
import pymupdf
from tqdm.auto import tqdm

class PDFTextExtractor:
    """Service for extracting text from PDF files."""

    @staticmethod
    def _format_text(text: str) -> str:
        """Formats text by replacing newlines and stripping whitespace."""
        return text.replace("\n", " ").strip()

    def extract(self, pdf_path: str) -> list[dict]:
        """Extracts text from a single PDF and returns a list of page-level text data."""
        all_pages_text = []

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        print(f"\n[INFO] Extracting text from: {pdf_path}")
        doc = pymupdf.open(pdf_path)

        for page_number, page in enumerate(tqdm(doc, desc=f"Processing {os.path.basename(pdf_path)}")):
            text = page.get_text()
            formatted_text = self._format_text(text)

            all_pages_text.append({
                "pdf_file": os.path.basename(pdf_path),
                "page_number": page_number,
                "page_char_count": len(formatted_text),
                "page_word_count": len(formatted_text.split(" ")),
                "page_sentence_count_raw": len(formatted_text.split(". ")),
                "page_token_count": len(formatted_text) / 4,  # Approximate token count
                "text": formatted_text
            })

        return all_pages_text

if __name__ == "__main__":
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "data", "sample-service-manual 1.pdf")
    
    try:
        extractor = PDFTextExtractor()
        pages_and_text = extractor.extract(pdf_path)
        print(pages_and_text[:2])
    except Exception as e:
        print(f"Error: {e}")
