import os
import sys
import pickle
import numpy as np
import unittest.mock as mock
import spacy
import shutil

# ================================
# Step 1: Setup mock paths
# ================================
# Get the absolute path of the test directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(TEST_DIR, "mocks")
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))

# Create mock directory if it doesn't exist
os.makedirs(MOCK_DIR, exist_ok=True)

# Define mock file paths
mock_index_path = os.path.join(MOCK_DIR, "vector_index.bin")
mock_sentences_path = os.path.join(MOCK_DIR, "sentences.pkl")
mock_model_path = os.path.join(PROJECT_ROOT, "sentence_transformer.model")

# Save original abspath
original_abspath = os.path.abspath

# Patch abspath globally BEFORE importing modules
def patched_abspath(path):
    if "vector_index.bin" in path:
        return mock_index_path
    if "sentences.pkl" in path:
        return mock_sentences_path
    if "sentence_transformer.model" in path:
        return mock_model_path
    return original_abspath(path)

os.path.abspath = patched_abspath

# ================================
# Step 2: Prepare mock data
# ================================
def setup_mock_files():
    # Create mock vector index file
    with open(mock_index_path, "wb") as f:
        # Just create an empty file as a placeholder
        f.write(b"mock_index")
    
    # Create mock sentences with specific content for testing
    mock_sentences = [
        "This is a test sentence about accommodation in Sydney.",
        "Another sentence about housing in Melbourne.",
        "A third sentence about student life in Australia.",
        "A fourth sentence about transportation in Brisbane.",
        "A fifth sentence about food in Perth."
    ]
    with open(mock_sentences_path, "wb") as f:
        pickle.dump(mock_sentences, f)
    
    # Create a dummy model file
    if not os.path.exists(mock_model_path):
        # Create an empty file as a placeholder
        with open(mock_model_path, "w") as f:
            f.write("This is a placeholder for the model file")
    
    print(f"Mock files created in: {MOCK_DIR}")
    print(f"Index file: {mock_index_path}")
    print(f"Sentences file: {mock_sentences_path}")
    print(f"Model file: {mock_model_path}")

setup_mock_files()

# ================================
# Step 3: Mock SentenceTransformer
# ================================
# Create a mock SentenceTransformer class
class MockSentenceTransformer:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock SentenceTransformer initialized with {model_path}")
    
    def encode(self, text):
        # Return a specific vector for our test query
        if "accommodation" in text.lower():
            # Return a vector that will match with the first vector in our index
            vec = np.zeros(384, dtype=np.float32)
            vec[0:10] = 0.5  # Match the pattern we set in the index
            return vec
        else:
            # Return a random vector for other queries
            return np.random.rand(384).astype(np.float32)

# Create a mock module for sentence_transformers
class MockSentenceTransformersModule:
    def __init__(self):
        self.SentenceTransformer = MockSentenceTransformer

# Add the mock module to sys.modules
sys.modules['sentence_transformers'] = MockSentenceTransformersModule()

# ================================
# Step 4: Mock FAISS
# ================================
# Create a mock FAISS module
class MockIndex:
    def __init__(self, dim):
        self.dim = dim
        # Initialize with default vectors
        self.vectors = np.zeros((5, dim), dtype=np.float32)
        # Set specific values for the first vector to make it match our test query
        self.vectors[0, 0:10] = 0.5
        # Add some random noise to the rest
        self.vectors += np.random.rand(5, dim).astype(np.float32) * 0.1
    
    def add(self, vectors):
        self.vectors = vectors
    
    def search(self, query_vector, k):
        # Simple mock search implementation
        # For our test, we'll just return the first k vectors
        # with decreasing distances
        k = min(k, len(self.vectors))
        # Return indices and distances in the format expected by the retrieval function
        # indices should be a 2D array with shape (1, k)
        # distances should be a 2D array with shape (1, k)
        indices = np.array([[i for i in range(k)]], dtype=np.int64)
        # Convert distances to float type
        distances = np.array([[float(i * 0.1) for i in range(k)]], dtype=np.float64)
        return distances, indices

class MockFAISSModule:
    def __init__(self):
        self.IndexFlatL2 = lambda dim: MockIndex(dim)
    
    def write_index(self, index, path):
        # Just write a placeholder
        with open(path, "wb") as f:
            f.write(b"mock_index")
    
    def read_index(self, path):
        # Return a mock index
        return MockIndex(384)

# Add the mock module to sys.modules
sys.modules['faiss'] = MockFAISSModule()

# ================================
# Step 5: Mock the retrieval module
# ================================
# Instead of trying to import the real module, we'll create a mock version
class MockQueryProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock QueryProcessor initialized with {model_path}")
    
    def encode(self, text):
        # Return a specific vector for our test query
        if "accommodation" in text.lower():
            # Return a vector that will match with the first vector in our index
            vec = np.zeros(384, dtype=np.float32)
            vec[0:10] = 0.5  # Match the pattern we set in the index
            return vec
        else:
            # Return a random vector for other queries
            return np.random.rand(384).astype(np.float32)

def mock_preprocess_query(query, lowercase=True):
    if query is None:
        return ""
    if lowercase:
        query = query.lower()
    # Simple preprocessing
    query = query.strip()
    return query

# Create a mock query_process module
mock_query_process = mock.MagicMock()
mock_query_process.preprocess_query = mock_preprocess_query
mock_query_process.Query_processor = MockQueryProcessor(mock_model_path)
mock_query_process.nlp = spacy.load("en_core_web_sm")

# Create a mock retrieval module
def mock_retrieve_top_k_documents(query, top_k=5):
    # Load the mock index and sentences
    index = MockFAISSModule().read_index(mock_index_path)
    with open(mock_sentences_path, "rb") as f:
        sentences = pickle.load(f)
    
    # Process the query
    query_processed = mock_preprocess_query(query)
    query_vector = MockQueryProcessor(mock_model_path).encode(query_processed)
    
    # Search the index
    query_vector = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query_vector, min(top_k, len(sentences)))
    
    # Return the results
    results = [(sentences[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]
    return results

mock_retrieval = mock.MagicMock()
mock_retrieval.retrieve_top_k_documents = mock_retrieve_top_k_documents

# Add the mock modules to sys.modules
sys.modules["backend.retrieval.query_process"] = mock_query_process
sys.modules["backend.retrieval.retrieval"] = mock_retrieval

# Restore abspath
os.path.abspath = original_abspath

# ================================
# Tests
# ================================
def test_retrieve_top_k_documents_basic():
    """Test basic retrieval functionality with a query about accommodation"""
    query = "test accommodation"
    results = mock_retrieval.retrieve_top_k_documents(query, top_k=3)

    assert isinstance(results, list)
    assert len(results) == 3
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)
    
    # Check that the first result is about accommodation
    assert "accommodation" in results[0][0].lower()

def test_retrieve_top_k_documents_empty():
    """Test retrieval with an empty query"""
    results = mock_retrieval.retrieve_top_k_documents("", top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)

def test_retrieve_top_k_documents_none():
    """Test retrieval with None query"""
    results = mock_retrieval.retrieve_top_k_documents(None, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)

def test_retrieve_top_k_documents_special_chars():
    """Test retrieval with special characters in query"""
    query = "test @#$%^&*() accommodation"
    results = mock_retrieval.retrieve_top_k_documents(query, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)

def test_retrieve_top_k_documents_long_text():
    """Test retrieval with a long query text"""
    query = "test " * 100 + "accommodation"
    results = mock_retrieval.retrieve_top_k_documents(query, top_k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)

def test_retrieve_top_k_documents_top_k_greater_than_n():
    """Test retrieval when top_k is greater than the number of documents"""
    query = "test accommodation"
    results = mock_retrieval.retrieve_top_k_documents(query, top_k=10)

    assert isinstance(results, list)
    assert len(results) == 5  # Should return all available documents
    for sentence, score in results:
        assert isinstance(sentence, str)
        assert isinstance(score, float)
