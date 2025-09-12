from vertexai.language_models import TextEmbeddingModel
import vertexai

def get_embedding(text, project_id, location="us-central1"):
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

    # Initialize the embedding model
    embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

    # Get embeddings for the input text
    embeddings = embedding_model.get_embeddings([text])

    return embeddings[0].values