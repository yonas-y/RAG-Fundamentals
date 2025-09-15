import vertexai
from vertexai.language_models import TextEmbeddingModel
from app.config import EMBEDDING_MODEL, PROJECT_ID, LOCATION

def get_embedding(text: str) -> list[float]:
    """
    Generates a text embedding using a Vertex AI model.

    Args:
        text (str): The input string to embed.

    Returns:
        A list of floats representing the embedding vector.
    """
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location=LOCATION)

   # Load embedding model (Gemini-based)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    # Get embeddings
    embeddings = model.get_embeddings([text])[0].values

    return embeddings