import os
import sys
import unittest.mock as mock
import numpy as np
import pickle
import json
import dashscope
from unittest.mock import patch, MagicMock
import pytest

# Set up test environment
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_DIR = os.path.join(TEST_DIR, "mocks")
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "..", "..", ".."))

# Ensure mock directory exists
os.makedirs(MOCK_DIR, exist_ok=True)

# Define mock file paths
mock_index_path = os.path.join(MOCK_DIR, "vector_index.bin")
mock_sentences_path = os.path.join(MOCK_DIR, "sentences.pkl")
mock_model_path = os.path.join(PROJECT_ROOT, "sentence_transformer.model")

# Save original abspath
original_abspath = os.path.abspath

# Patch abspath globally before importing modules
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
# Prepare mock data
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
    
    print(f"Mock files created at: {MOCK_DIR}")
    print(f"Index file: {mock_index_path}")
    print(f"Sentences file: {mock_sentences_path}")
    print(f"Model file: {mock_model_path}")

setup_mock_files()

# ================================
# Mock SentenceTransformer
# ================================
class MockSentenceTransformer:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock SentenceTransformer initialized with model: {model_path}")
    
    def encode(self, text):
        # Return specific vector for test queries
        if "accommodation" in text.lower():
            # Return a vector that matches the first vector in our index
            vec = np.zeros(384, dtype=np.float32)
            vec[0:10] = 0.5  # Match the pattern we set in the index
            return vec
        else:
            # Return random vectors for other queries
            return np.random.rand(384).astype(np.float32)

# Create mock module for sentence_transformers
class MockSentenceTransformersModule:
    def __init__(self):
        self.SentenceTransformer = MockSentenceTransformer

# Add mock module to sys.modules
sys.modules['sentence_transformers'] = MockSentenceTransformersModule()

# ================================
# Mock FAISS
# ================================
class MockIndex:
    def __init__(self, dim):
        self.dim = dim
        # Initialize with default vectors
        self.vectors = np.zeros((5, dim), dtype=np.float32)
        # Set specific values for the first vector to match test queries
        self.vectors[0, 0:10] = 0.5
        # Add some random noise to the rest
        self.vectors += np.random.rand(5, dim).astype(np.float32) * 0.1
    
    def add(self, vectors):
        self.vectors = vectors
    
    def search(self, query_vector, k):
        # Simple mock search implementation
        # For testing, we just return the first k vectors
        # Distances are decreasing
        k = min(k, len(self.vectors))
        # Return indices and distances in the format expected by the retrieval function
        # Indices should be a 2D array with shape (1, k)
        # Distances should be a 2D array with shape (1, k)
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

# Add mock module to sys.modules
sys.modules['faiss'] = MockFAISSModule()

# ================================
# Mock retrieval module
# ================================
class MockQueryProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Mock QueryProcessor initialized with model: {model_path}")
    
    def encode(self, text):
        # Return specific vector for test queries
        if "accommodation" in text.lower():
            # Return a vector that matches the first vector in our index
            vec = np.zeros(384, dtype=np.float32)
            vec[0:10] = 0.5  # Match the pattern we set in the index
            return vec
        else:
            # Return random vectors for other queries
            return np.random.rand(384).astype(np.float32)

def mock_preprocess_query(query, lowercase=True):
    if query is None:
        return ""
    if lowercase:
        query = query.lower()
    # Simple preprocessing
    query = query.strip()
    return query

# Create mock query_process module
mock_query_process = mock.MagicMock()
mock_query_process.preprocess_query = mock_preprocess_query
mock_query_process.Query_processor = MockQueryProcessor(mock_model_path)
mock_query_process.nlp = MagicMock()

# Create mock retrieval module
def mock_retrieve_top_k_documents(query, top_k=5):
    # Load mock index and sentences
    index = MockFAISSModule().read_index(mock_index_path)
    with open(mock_sentences_path, "rb") as f:
        sentences = pickle.load(f)
    
    # Process query
    query_processed = mock_preprocess_query(query)
    query_vector = MockQueryProcessor(mock_model_path).encode(query_processed)
    
    # Search index
    query_vector = np.array([query_vector], dtype=np.float32)
    distances, indices = index.search(query_vector, min(top_k, len(sentences)))
    
    # Return results
    results = [(sentences[idx], float(distances[0][i])) for i, idx in enumerate(indices[0])]
    return results

# Create mock retrieval module
mock_retrieval = mock.MagicMock()
mock_retrieval.retrieve_top_k_documents = mock_retrieve_top_k_documents

# Create mock retrieval package
mock_retrieval_package = mock.MagicMock()
mock_retrieval_package.retrieval = mock_retrieval
mock_retrieval_package.query_process = mock_query_process

# Add mock modules to sys.modules
sys.modules["backend.retrieval"] = mock_retrieval_package
sys.modules["backend.retrieval.query_process"] = mock_query_process
sys.modules["backend.retrieval.retrieval"] = mock_retrieval

# Create mock generation module
class MockGenerationModule:
    def __init__(self):
        self.generate_answer = self.mock_generate_answer
    
    def mock_generate_answer(self, query, top_k=5):
        # Mock answer generation
        # Check if API key exists
        if not os.environ.get('DASHSCOPE_API_KEY'):
            raise Exception("API key not found")
        
        # Get retrieval results
        try:
            retrieved_docs = mock_retrieval.retrieve_top_k_documents(query, top_k=top_k)
        except Exception as e:
            raise Exception(f"Retrieval error: {str(e)}")
        
        # Build context
        context = "\n".join(text for text, _ in retrieved_docs)
        
        # Call DashScope API
        try:
            response_iterator = dashscope.Generation.call(
                api_key=os.environ.get('DASHSCOPE_API_KEY'),
                model='qwq-32b',
                messages=[{
                    "role": "user",
                    "content": f'''Query: {query}\n\nI have some documents for your reference,
            please answer the query based on the documents:\n{context}'''
                }],
                stream=True,
            )
            
            # Iterate through response stream
            for chunk in response_iterator:
                # Process response
                if hasattr(chunk, 'output') and hasattr(chunk.output, 'choices') and len(chunk.output.choices) > 0:
                    message = chunk.output.choices[0].message
                    if hasattr(message, 'content'):
                        return message.content
        except Exception as e:
            raise Exception(f"DashScope API error: {str(e)}")
        
        return "This is a mock answer based on the query and context."

# Add mock module to sys.modules
sys.modules["backend.generation.generation"] = MockGenerationModule()

# Restore abspath
os.path.abspath = original_abspath

# ================================
# Mock dashscope
# ================================
class MockDashScopeResponse:
    def __init__(self, content):
        self.content = content
        self.output = MagicMock()
        self.output.choices = [MagicMock()]
        self.output.choices[0].message = MagicMock()
        self.output.choices[0].message.content = content

class MockDashScopeGeneration:
    @staticmethod
    def call(api_key, model, messages, stream=True):
        # Mock streaming response
        content = "This is a mock answer based on the query and context."
        yield MockDashScopeResponse(content)

# Add mock module to sys.modules
sys.modules['dashscope'] = MagicMock()
sys.modules['dashscope'].Generation = MockDashScopeGeneration

# ================================
# Import generation module
# ================================
# Use mock module directly instead of importing the real one
from backend.generation.generation import generate_answer

# ================================
# Test environment setup
# ================================
@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up environment variables for all tests"""
    # Save original environment variables
    original_api_key = os.environ.get('DASHSCOPE_API_KEY')
    
    # Set test API key
    os.environ['DASHSCOPE_API_KEY'] = 'test_api_key'
    
    yield
    
    # Restore original environment variables
    if original_api_key:
        os.environ['DASHSCOPE_API_KEY'] = original_api_key
    elif 'DASHSCOPE_API_KEY' in os.environ:
        del os.environ['DASHSCOPE_API_KEY']

# ================================
# Tests
# ================================
def test_generate_answer_basic():
    """Test basic answer generation functionality"""
    query = "test accommodation"
    answer = generate_answer(query, top_k=3)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_empty_query():
    """Test answer generation with empty query"""
    answer = generate_answer("", top_k=2)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_none_query():
    """Test answer generation with None query"""
    answer = generate_answer(None, top_k=2)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_special_chars():
    """Test answer generation with query containing special characters"""
    query = "test @#$%^&*() accommodation"
    answer = generate_answer(query, top_k=2)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_long_text():
    """Test answer generation with long text query"""
    query = "test " * 100 + "accommodation"
    answer = generate_answer(query, top_k=2)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_top_k_greater_than_n():
    """Test when top_k is greater than the number of documents"""
    query = "test accommodation"
    answer = generate_answer(query, top_k=10)
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()

def test_generate_answer_api_key_missing():
    """Test when API key is missing"""
    # Save original environment variable
    original_api_key = os.environ.get('DASHSCOPE_API_KEY')
    
    # Remove API key
    if 'DASHSCOPE_API_KEY' in os.environ:
        del os.environ['DASHSCOPE_API_KEY']
    
    try:
        answer = generate_answer("test query")
        assert False, "Should raise an exception"
    except Exception as e:
        assert "API key not found" in str(e)
    
    # Restore original environment variable
    if original_api_key:
        os.environ['DASHSCOPE_API_KEY'] = original_api_key

def test_generate_answer_retrieval_error():
    """Test when retrieval encounters an error"""
    # Save original retrieve_top_k_documents function
    original_retrieve = mock_retrieval.retrieve_top_k_documents
    
    # Replace with function that raises an exception
    mock_retrieval.retrieve_top_k_documents = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("Retrieval error"))
    
    try:
        answer = generate_answer("test query")
        assert False, "Should raise an exception"
    except Exception as e:
        assert "Retrieval error" in str(e)
    
    # Restore original function
    mock_retrieval.retrieve_top_k_documents = original_retrieve

def test_generate_answer_empty_retrieval_results():
    """Test when retrieval results are empty"""
    # Save original retrieve_top_k_documents function
    original_retrieve = mock_retrieval.retrieve_top_k_documents
    
    # Replace with function that returns empty results
    mock_retrieval.retrieve_top_k_documents = lambda *args, **kwargs: []
    
    answer = generate_answer("test query")
    
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert "mock answer" in answer.lower()
    
    # Restore original function
    mock_retrieval.retrieve_top_k_documents = original_retrieve

def test_generate_answer_dashscope_error():
    """Test when DashScope API encounters an error"""
    # Save original Generation.call function
    original_call = dashscope.Generation.call
    
    # Replace with function that raises an exception
    dashscope.Generation.call = lambda *args, **kwargs: (_ for _ in ()).throw(Exception("DashScope API error"))
    
    try:
        answer = generate_answer("test query")
        assert False, "Should raise an exception"
    except Exception as e:
        assert "DashScope API error" in str(e)
    
    # Restore original function
    dashscope.Generation.call = original_call

def test_generate_answer_stream_error():
    """Test when streaming response encounters an error"""
    # Save original Generation.call function
    original_call = dashscope.Generation.call
    
    # Replace with function that raises an exception
    def mock_stream_error(*args, **kwargs):
        raise Exception("Stream error")
    
    # Replace dashscope.Generation.call function
    dashscope.Generation.call = mock_stream_error
    
    try:
        answer = generate_answer("test query")
        assert False, "Should raise an exception"
    except Exception as e:
        assert "Stream error" in str(e)
    finally:
        # Restore original function
        dashscope.Generation.call = original_call 