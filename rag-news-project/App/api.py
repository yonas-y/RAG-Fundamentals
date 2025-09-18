# FastAPI endpoints for RAG.

from fastapi import FastAPI
from pydantic import BaseModel
from app.vector_store import VectorStore
from app.rag_pipeline import rag_query
from app.data_loader import load_documents, chunk_documents
from app.embedder import get_embedding
from app.config import doc_dir_path

app = FastAPI(title="RAG News API", version="1.0")

# Load or build vector store at startup
vs = None

@app.on_event("startup")
def startup_event():
    global vs
    vs = VectorStore.load()
    if not vs:
        print("⚠️  No existing vector store found, creating one...")
        documents = load_documents(doc_dir_path)
        chunks = chunk_documents(documents)
        vs = VectorStore(dim=768)
        for chunk in chunks:
            embedding = get_embedding(chunk.page_content)
            vs.add(embedding, chunk.page_content)
        vs.save()
        print("✅ Vector store built and saved.")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/query")
def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline with a user question.
    """
    answer = rag_query(request.query, vs)
    return {"query": request.query, "answer": answer}


@app.get("/health")
def health_check():
    return {"status": "ok", "vector_store_loaded": vs is not None}


@app.post("/rebuild")
def rebuild_vector_store():
    """
    Rebuild the vector store from the data directory on demand.
    Replaces the in-memory store and saves the new store to disk.
    """
    global vs
    try:
        print("🔁 Rebuilding vector store...")
        documents = load_documents(doc_dir_path)
        chunks = chunk_documents(documents)
        new_vs = VectorStore(dim=768)
        for chunk in chunks:
            embedding = get_embedding(chunk.page_content)
            new_vs.add(embedding, chunk.page_content)
        new_vs.save()
        vs = new_vs
        print(f"✅ Rebuilt vector store with {len(chunks)} chunks.")
        return {"status": "ok", "message": "Vector store rebuilt", "num_chunks": len(chunks)}
    except Exception as e:
        print(f"❌ Error rebuilding vector store: {e}")
        return {"status": "error", "message": str(e)}
