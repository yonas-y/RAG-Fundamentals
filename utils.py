from pathlib import Path

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
