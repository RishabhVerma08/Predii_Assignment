# Vehicle Spec RAG

A Retrieval-Augmented Generation (RAG) system designed to extract structured vehicle specifications (torque values, dimensions, etc.) from PDF service manuals. The system uses a specialized pipeline to process technical documents and provides an interactive chat interface for querying them.

##  Key Features

- **PDF Ingestion Pipeline**: Extracts, chunks, and indexes text from technical service manuals.
- **RAG Architecture**: Retrieves relevant context from vector storage to ground LLM responses.
- **Interactive UI**: Clean, responsive web interface for chatting with your manuals.
- **Manual Management**: drag-and-drop upload functionality to replace and index new manuals instantly.
- **Dynamic Prompting**: Custom "System Prompt" editor allowing users to tweak extraction rules and output formats on the fly.
- **Structured Output**: Designed to return precise JSON data for specifications (Component, Value, Unit).

##  Technology Stack

### Backend
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (High-performance Async API)
- **Language**: Python 3.10+
- **PDF Processing**: [PyMuPDF](https://pymupdf.readthedocs.io/) (Text extraction) & [SpaCy](https://spacy.io/) (Sentence splitting)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/) (Local, embedded vector search)
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-mpnet-base-v2`)
- **LLM**: [Google Gemini Pro](https://ai.google.dev/) (via `google-generativeai`)

### Frontend
- **Core**: Vanilla HTML/CSS/JavaScript (No complex build step required)
- **Design**: Responsive, modern dark-themed UI with real-time feedback.

---

## System Design

The application follows a modular Service-Oriented Architecture:

1.  **Ingestion Service**: 
    - **Extract**: text is pulled from PDFs using `PDFTextExtractor`.
    - **Chunk**: `TextChunker` splits text into semantic chunks (using SpaCy sentences) to preserve context.
    - **Embed**: `EmbeddingService` converts chunks into dense vector representations.
    - **Store**: Vectors and metadata are stored in `ChromaDB`.

2.  **Retrieval Service**:
    - Queries the Vector Store using cosine similarity to find the most relevant chunks for a user question.

3.  **Generation Service**:
    - Constructs a prompt using the retrieved context and a persistent template (`config/prompt_template.txt`).
    - Sends the payload to Google Gemini to generate a structured JSON response.

---

##  Getting Started

1.  **Prerequisites**:
    - Python 3.10+ installed.
    - A specific PDF manual (placed in `data/` or uploaded via UI).
    - A Google Gemini API Key.

2.  **Installation**:
    ```bash
    # Clone repository
    git clone <repo-url>
    cd vehicle-spec-rag

    # Create virtual environment
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

    # Install dependencies
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

4.  **Running the App**:
    ```bash
    # Start the FastAPI server
    python vehicle-spec-rag/app.py
    ```
    Access the UI at `http://localhost:3000`.

---

##  Ideas for Improvement

1.  **Hybrid Search (Semantic + Keyword)**:
    - Currently relies on dense vector retrieval. Combining this with BM25 (keyword search) would improve accuracy for exact part numbers or specific torque values that might not have strong semantic meaning.

2.  **Recursive chunking / Parent Document Retrieval**:
    - Break documents into smaller child chunks for retrieval but return the larger parent chunk to the LLM to provide more surrounding context.

3.  **Multi-Modal RAG**:
    - Technical manuals have diagrams. Using a multi-modal embedding model (like CLIP or Gemini Pro Vision) to index diagrams and tables would allow users to ask "Show me the diagram for the caliper bolt".

4.  **Metadata Filtering**:
    - Extract more metadata (Year, Make, Model, Section) during ingestion to allow filtered queries.

5.  **Streaming Responses**:
    - Implement Server-Sent Events (SSE) to stream the LLM's response token-by-token for a faster perceived latency.

6.  **Response Caching**:
    - Cache frequent queries (like "What is the oil capacity?") to avoid re-generating answers and save API costs.
