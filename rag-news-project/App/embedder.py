# Generate embeddings with Vertex AI

from google.cloud import aiplatform

def get_embedding(text, project_id, location="us-central1"):
    aiplatform.init(project=project_id, location=location)
    model = aiplatform.TextEmbeddingModel.from_pretrained("textembedding-gecko")
    return model.get_embeddings([text])[0].values
