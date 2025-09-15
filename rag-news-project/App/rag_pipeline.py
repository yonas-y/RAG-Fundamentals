# Core RAG workflows!
from google import genai
from google.genai.types import HttpOptions
import vertexai
from app.embedder import get_embedding

from app.config import PROJECT_ID, LOCATION

def rag_query(user_query, vector_store):
    # Embed query
    query_embedding = get_embedding(user_query)

    # Retrieve top docs
    results = vector_store.search(query_embedding, k=3)
    context = "\n".join([doc for doc, _ in results])

    # Initialize Vertex AI (ensure ADC or service account credentials are set)
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Initialize Gen AI client using Vertex AI context
    client = genai.Client(
        vertexai=vertexai,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(api_version="v1"),
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Answer based on:\n{context}\n\nQuestion: {user_query}",
    )

    return response.text
