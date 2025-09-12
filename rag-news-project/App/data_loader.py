# Load & chunk documents!

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(data_path="data/"):
    docs = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def chunk_documents(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(docs)
