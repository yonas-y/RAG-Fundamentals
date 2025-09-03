from pathlib import Path
from google.cloud import aiplatform

def load_documents(directory: str, encoding: str = "utf-8") -> list[dict]:
    documents = []
    directory_path = Path(directory)

    for file_path in directory_path.glob("*.txt"):
        try:
            text = file_path.read_text(encoding=encoding)
            documents.append({
                "content": text,
                "filename": file_path.name,
                "path": str(file_path.resolve()),
                "size": file_path.stat().st_size
            })
        except Exception as e:
            print(f"⚠️ Skipping {file_path.name} due to error: {e}")

    return documents


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]

        # optional: avoid cutting off mid-word
        if end < text_length and not text[end].isspace():
            last_space = chunk.rfind(" ")
            if last_space != -1:
                chunk = chunk[:last_space]
                end = start + last_space

        chunks.append(chunk.strip())
        start = end - overlap  # move forward with overlap

    return chunks


def chunk_documents(
        documents: list[dict], 
        chunk_size: int = 500, 
        overlap: int = 50) -> list[dict]:
    """Split documents into chunks with metadata, ready for RAG."""
    chunked_docs = []

    for doc in documents:
        for i, chunk in enumerate(split_text(doc["content"], chunk_size, overlap)):
            chunked_docs.append({
                "content": chunk,
                "filename": doc["filename"],
                "path": doc["path"],
                "doc_size": doc["size"],
                "chunk_id": i,
                "chunk_size": len(chunk),
            })

    return chunked_docs


def embed_chunks(
        chunks: list[dict], 
        embedding_model: str = "textembedding-gecko@001", 
        location: str = "us-central1", 
        project: str = None) -> list[dict]:
    """
    Generate embeddings for a list of chunk dicts using Vertex AI.
    
    Args:
        chunks: List of dicts with at least 'content' key.
        embedding_model: Vertex AI embedding model name.
        location: Vertex AI region.
        project: Google Cloud project ID (optional, uses default if None).
    
    Returns:
        List of dicts, each with original metadata + 'embedding' vector.
    """
    aiplatform.init(project=project, location=location)
    embeddings_client = aiplatform.gapic.PredictionServiceClient()

    embedded_chunks = []
    endpoint = f"projects/{project}/locations/{location}/publishers/google/models/{embedding_model}/endpoint"

    for chunk in chunks:
        # Vertex AI expects a list of text inputs
        response = embeddings_client.predict(
            endpoint=endpoint,
            instances=[{"content": chunk["content"]}],
        )
        # Extract embedding vector
        embedding_vector = response.predictions[0]["embedding"]

        # Append embedding to chunk dict
        new_chunk = chunk.copy()
        new_chunk["embedding"] = embedding_vector
        embedded_chunks.append(new_chunk)

    return embedded_chunks
