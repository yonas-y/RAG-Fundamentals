# Import packages!
import os
from dotenv import load_dotenv
from google.adk import generativeai as genai
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")    

# Embeddings using Google API and ChromaDB.


# Initialize Chroma client with persistence.


# Create a client.


# A function to load all the files from a directory.
def load_documents(directory: str) -> list[str]:
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r") as file:
                documents.append(file.read())

    return documents

# Function to split text into chunks.
def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Load documents from the specified directory.
doc_dir_path = "./data/news_articles/"
documents = load_documents(doc_dir_path)

# Split each document into chunks.
chunked_documents = [split_text(doc) for doc in documents]

