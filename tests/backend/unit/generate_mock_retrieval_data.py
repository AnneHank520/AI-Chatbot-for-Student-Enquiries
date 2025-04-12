import os
import pickle
import faiss
import numpy as np

# Ensure directory exists
mock_dir = "tests/backend/unit/mocks"
os.makedirs(mock_dir, exist_ok=True)

# Create a dummy sentence list
sentences = ["Doc 1", "Doc 2", "Doc 3"]
with open(os.path.join(mock_dir, "sentences.pkl"), "wb") as f:
    pickle.dump(sentences, f)

# Create a dummy FAISS index
dim = 384  # Make sure this matches your SentenceTransformer output dimension
index = faiss.IndexFlatL2(dim)

# Add dummy vectors (3 vectors for 3 sentences)
vectors = np.random.rand(3, dim).astype("float32")
index.add(vectors)

faiss.write_index(index, os.path.join(mock_dir, "vector_index.bin"))

print("Mock FAISS index and sentence list created.")
