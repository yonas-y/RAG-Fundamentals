# Store & retrieve embeddings
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.vectors = []
        self.docs = []

    def add(self, embedding, doc):
        self.index.add(np.array([embedding]).astype("float32"))
        self.vectors.append(embedding)
        self.docs.append(doc)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(np.array([query_embedding]).astype("float32"), k)
        return [(self.docs[i], distances[0][j]) for j, i in enumerate(indices[0])]
