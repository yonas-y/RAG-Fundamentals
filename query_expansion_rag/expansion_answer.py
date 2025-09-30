# Import Packages!
from dotenv import load_dotenv
import os

# Load environment variables from .env file!
load_dotenv()

vertexai_key = os.getenv("VERTEXAI_KEY")

# Read the files from the data directory!


# Split the text into smaller chunks!
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    SentenceTransformersTokenTextSplitter
    )

# Initialize the text splitter!
# Initialize a token text splitter!

# Embed the tokens using the SentenceTransformers model and create a collection!

# Create the query and look through the collection!

# Create an augmented quesry generator and use an LLM for a response!
def augmented_query_generator(query, model_name="gemini-1.5-flash"):
    # This function generates an augmented query using a language model.

    return None


original_query = ""
hypothetical_response = augmented_query_generator(original_query)

joint_query = f"{original_query} {hypothetical_response}"

# Use the joint query to query the collection and retrieve documents!
# (include documnents and embeddings in the response!)

# Plot to see the original query and the augmented query in the embedding space!








