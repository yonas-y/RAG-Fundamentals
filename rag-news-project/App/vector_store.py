# Store & retrieve embeddings with FAISS persistence
import faiss
import numpy as np
import pickle
import os
from app.config import PERSIST_DIR

INDEX_FILE = os.path.join(PERSIST_DIR, "news_faiss.index")
DOCS_FILE = os.path.join(PERSIST_DIR, "news_docs.pkl")

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, embedding, doc):
        self.index.add(np.array([embedding]).astype("float32"))
        self.docs.append(doc)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        return [(self.docs[i], distances[0][j]) for j, i in enumerate(indices[0])]

    def save(self):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(self.docs, f)

    @classmethod
    def load(cls):
        if not os.path.exists(INDEX_FILE) or not os.path.exists(DOCS_FILE):
            return None
        index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, "rb") as f:
            docs = pickle.load(f)
        store = cls(index.d)
        store.index = index
        store.docs = docs
        return store
    