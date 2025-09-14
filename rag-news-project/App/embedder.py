from vertexai.language_models import TextEmbeddingModel
import vertexai
from app.config import EMBEDDING_MODEL

def get_embedding(text: str, project_id: str, location: str = "us-central1") -> list[float]:
    """
    Generates a text embedding using a Vertex AI model.

    Args:
        text (str): The input string to embed.
        project_id (str): The Google Cloud project ID.
        location (str): The location of the model (e.g., "us-central1").

    Returns:
        A list of floats representing the embedding vector.
    """
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

   # Load embedding model (Gemini-based)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    # Get embeddings
    embeddings = model.get_embeddings([text])

    return embeddings[0].values