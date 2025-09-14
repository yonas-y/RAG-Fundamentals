# Core RAG workflows!
from google.cloud import aiplatform
from app.embedder import get_embedding

from app.config import EMBEDDING_MODEL

def rag_query(user_query, vector_store, project_id):
    # Embed query
    query_embedding = get_embedding(user_query, project_id)

    # Retrieve top docs
    results = vector_store.search(query_embedding, k=3)
    context = "\n".join([doc for doc, _ in results])

    # Generate response
    aiplatform.init(project=project_id)
    model = aiplatform.GenerativeModel(EMBEDDING_MODEL)
    response = model.generate_content(f"Answer based on:\n{context}\n\nQuestion: {user_query}")
    
    return response.text
