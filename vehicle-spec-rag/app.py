
import os
import uvicorn
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import shutil

from vectorstore.chroma_db import ChromaDBService
from vectorstore.embeddings import EmbeddingService
from vectorstore.retriever import Retriever
from services.ingestion import IngestionService
from llm.gemini_client import GeminiClient
from llm.prompt_formatter import prompt_formatter_gemini

# --- Configuration ---
HOST = "0.0.0.0"
PORT = 3000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, "data", "chroma_store")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# --- Global Services ---
# Initialize globally to reuse across requests
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    print("[INFO] Starting up API...")
    try:
        services["embedder"] = EmbeddingService()
        services["chroma"] = ChromaDBService(persist_directory=CHROMA_DB_PATH)
        services["retriever"] = Retriever(services["chroma"], services["embedder"])
        services["ingestion"] = IngestionService(services["chroma"], services["embedder"])
        services["llm_client"] = GeminiClient()
        print("[INFO] Services initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        # raise e 
        # Don't raise in dev to allow server to start partially if needed, 
        # but for this assignment, better to see error. 
        # Actually, if we raise, uvicorn might keep restarting. 
        # Printing error is enough for now. The endpoint checks for services.
    
    yield
    
    print("[INFO] Shutting down API...")
    services.clear()

app = FastAPI(title="Vehicle Spec RAG API", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# --- Data Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: dict | list 

# --- Endpoints ---

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/query", response_model=QueryResponse)
def query_specs(request: QueryRequest):
    if not services:
        raise HTTPException(status_code=500, detail="Services not initialized.")
    if "llm_client" not in services:
         raise HTTPException(status_code=500, detail="LLM Client failed to initialize.")

    query_text = request.query
    print(f"[API] Received query: {query_text}")
    
    try:
        # 1. Retrieve Context
        retriever: Retriever = services["retriever"]
        context_docs = retriever.retrieve(query_text, k=5)
        
        # Build context items dict
        context_items = [{"sentence_chunk": doc} for doc in context_docs]
        
        # 2. Format Prompt
        prompt = prompt_formatter_gemini(query_text, context_items)
        
        # 3. Generate Answer
        llm_client: GeminiClient = services["llm_client"]
        raw_response = llm_client.generate_content(prompt)
        
        # 4. Parse JSON
        clean_response = raw_response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        clean_response = clean_response.strip()
        
        try:
            answer_json = json.loads(clean_response)
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON: {clean_response}")
            answer_json = {
                "error": "Model response was not valid JSON", 
                "raw_response": clean_response
            }

        return QueryResponse(query=query_text, answer=answer_json)

    except Exception as e:
        print(f"[ERROR] Processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_manual(file: UploadFile = File(...)):
    if not services:
        raise HTTPException(status_code=500, detail="Services not initialized.")
    
    try:
        # Save file temporarily
        file_path = os.path.join(BASE_DIR, "data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"[API] Uploaded file: {file.filename}")
        
        # Trigger ingestion
        ingestion: IngestionService = services["ingestion"]
        num_chunks = ingestion.process_file(file_path)
        
        return JSONResponse(content={
            "status": "success", 
            "message": f"Successfully processed '{file.filename}'. Index rebuilt with {num_chunks} chunks."
        })
        
    except Exception as e:
        print(f"[ERROR] Upload processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host=HOST, port=PORT, reload=True)
