# Entry point (CLI or FastAPI)

from app.data_loader import load_documents, chunk_documents
from app.embedder import get_embedding
from app.vector_store import VectorStore
from app.rag_pipeline import rag_query
from app.config import doc_dir_path

if __name__ == "__main__":

    # Try to load an existing vector store
    vs = VectorStore.load()

    if not vs:
        print("⚠️ No existing vector store found, creating a new one...")

        # Load and embed docs
        documents = load_documents(doc_dir_path)
        print(f"Loaded {len(documents)} documents from {doc_dir_path}")
        chunks = chunk_documents(documents)
        print(f"Created {len(chunks)} chunks from the documents.")

        # Create new vector store
        vs = VectorStore(dim=768)  # Gecko embedding size
        for chunk in chunks:
            embedding = get_embedding(chunk.page_content)
            vs.add(embedding, chunk.page_content)
        print("Finished adding chunks to vector store.")

        # Save to disk for persistence
        vs.save()
        print("✅ Vector store saved to disk.")
        
    # Example user query
    query = "What are the latest news about Databricks?"
    print("\n🤖 Answer:\n", rag_query(query, vs))