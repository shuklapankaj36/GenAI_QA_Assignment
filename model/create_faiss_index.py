import json
import numpy as np
import faiss
import os
import pickle  # Import pickle for saving the index
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import

# Load your dataset
with open("./data/qa_dataset.json", "r") as f:
    data = json.load(f)

# Initialize the Hugging Face embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Generate embeddings for each context in the dataset
embeddings = []
for item in data:
    embedding = embedding_model.embed_query(item["context"])  # Updated method
    embeddings.append(embedding)

# Convert to a NumPy array
embeddings = np.array(embeddings).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for the dimensionality of the embeddings
index.add(embeddings)  # Add the embeddings to the index

# Create the directory if it doesn't exist
output_dir = "faiss_index"
os.makedirs(output_dir, exist_ok=True)  # Create the directory

# Save the index in FAISS binary format
faiss.write_index(index, os.path.join(output_dir, "index.faiss"))
