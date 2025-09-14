# Entry point (CLI or FastAPI)

from app.data_loader import load_documents, chunk_documents
from app.embedder import get_embedding
from app.vector_store import VectorStore

 
PROJECT_ID = "rag-fundamentals-471014"
LOCATION = "us-central1"
doc_dir_path = "data/"

if __name__ == "__main__":
    # Load and embed docs
    documents = load_documents(doc_dir_path)
    print(f"Loaded {len(documents)} documents from {doc_dir_path}")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks from the documents.")

    # Create vector store
    vs = VectorStore(dim=768)  # Gecko embedding size
    for chunk in chunks[:5]:
        embedding = get_embedding(chunk.page_content, PROJECT_ID, LOCATION)
        vs.add(embedding, chunk.page_content)
    print("Finished adding chunks to vector store.")

    