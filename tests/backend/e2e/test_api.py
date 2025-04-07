import os
import sys
import json
import pytest
import requests
import tempfile
import shutil
import logging
from pathlib import Path
from dotenv import load_dotenv
from requests.exceptions import RequestException, ConnectionError, Timeout

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Set DashScope API key
load_dotenv()
# get dashscope key from env
API_BASE_URL = "http://localhost:5000/api"

# Test data directories
TEST_DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "backend", "e2e", "test_data")
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Global variable to store the processed PDF filename
processed_pdf_filename = None

def make_request(method, url, **kwargs):
    """Helper function to make HTTP requests with error handling"""
    try:
        logger.info(f"Making {method} request to {url}")
        if 'files' in kwargs:
            logger.info(f"Uploading file: {list(kwargs['files'].keys())[0]}")
        if 'json' in kwargs:
            logger.info(f"Request payload: {kwargs['json']}")
        
        response = requests.request(method, url, timeout=300, **kwargs)
        
        logger.info(f"Response status code: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
        else:
            logger.info(f"Success response: {response.text[:200]}...")
        
        return response
    except ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise
    except Timeout as e:
        logger.error(f"Request timeout: {str(e)}")
        raise
    except RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def get_test_pdf():
    """Get the test PDF file"""
    pdf_path = os.path.join(PROJECT_ROOT, "uploads", "Resource.pdf")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        return None
    
    logger.info(f"Using PDF file: {pdf_path}")
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB
    logger.info(f"PDF file size: {file_size:.2f} MB")
    return pdf_path

def test_status_endpoint():
    """Test /api/status endpoint"""
    response = make_request('GET', f"{API_BASE_URL}/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "running"
    assert "vector_db_loaded" in data
    assert "model_loaded" in data
    assert "sentences_loaded" in data
    assert "dashscope_api_available" in data

@pytest.fixture(scope="module")
def process_pdf():
    """Fixture to process PDF file once for all tests"""
    global processed_pdf_filename
    
    pdf_path = get_test_pdf()
    if not pdf_path:
        pytest.skip("PDF file not found")
    
    logger.info("Starting PDF processing test")
    with open(pdf_path, "rb") as f:
        filename = os.path.basename(pdf_path)
        files = {"file": (filename, f, "application/pdf")}
        try:
            response = make_request('POST', f"{API_BASE_URL}/process-pdf", files=files)
            logger.info("PDF processing completed successfully")
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            raise
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "filename" in data
    assert data["filename"] == filename
    assert "sentences_added" in data
    assert "total_sentences" in data
    assert "vector_db_path" in data
    assert "url_count" in data
    assert "keywords_count" in data
    
    processed_pdf_filename = filename
    assert processed_pdf_filename is not None

def test_query_endpoint(process_pdf):
    """Test /api/query endpoint"""
    assert processed_pdf_filename is not None
    
    logger.info("Starting PDF query test")
    query_data = {
        "query": "test content",
        "top_k": 5,
        "context_size": 2
    }
    try:
        response = make_request('POST', f"{API_BASE_URL}/query", json=query_data)
        logger.info("PDF query completed successfully")
    except Exception as e:
        logger.error(f"PDF query failed: {str(e)}")
        raise
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "processed_query" in data
    assert "keywords" in data
    assert "results" in data
    assert "prompt_data" in data
    assert len(data["results"]) > 0

def test_generate_answer_endpoint(process_pdf):
    """Test /api/generate-answer endpoint"""
    assert processed_pdf_filename is not None
    
    logger.info("Starting PDF answer generation test")
    query_data = {
        "query": "test content",
        "top_k": 5,
        "context_size": 2,
        "model": "qwen-plus"
    }
    try:
        response = make_request('POST', f"{API_BASE_URL}/generate-answer", json=query_data)
        logger.info("PDF answer generation completed successfully")
    except Exception as e:
        logger.error(f"PDF answer generation failed: {str(e)}")
        raise
    
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "answer" in data
    assert "model" in data
    assert len(data["answer"]) > 0

def test_list_files_endpoint(process_pdf):
    """Test /api/list-files endpoint"""
    assert processed_pdf_filename is not None
    
    response = make_request('GET', f"{API_BASE_URL}/list-files")
    assert response.status_code == 200
    data = response.json()
    assert "files" in data
    assert processed_pdf_filename in data["files"]

def test_delete_pdf_content_endpoint(process_pdf):
    """Test /api/delete-pdf-content endpoint"""
    assert processed_pdf_filename is not None
    
    delete_data = {"filename": processed_pdf_filename}
    response = make_request('POST', f"{API_BASE_URL}/delete-pdf-content", json=delete_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "indices_removed" in data
    assert "sentences_remaining" in data
    assert "url_count" in data
    assert "keywords_count" in data

def test_error_handling():
    """Test error handling"""
    response = make_request('POST', f"{API_BASE_URL}/process-pdf")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    
    response = make_request('POST', f"{API_BASE_URL}/query", json={})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    
    # Test invalid PDF content
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(b"Invalid PDF content")
        temp_pdf.flush()
        files = {"file": ("invalid.pdf", open(temp_pdf.name, "rb"), "application/pdf")}
        response = make_request('POST', f"{API_BASE_URL}/process-pdf", files=files)
        assert response.status_code == 500
        data = response.json()
        assert "error" in data

def test_empty_pdf():
    """Test processing empty/blank PDF file"""
    # Create a minimal valid but empty PDF
    empty_pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000052 00000 n
0000000101 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
163
%%EOF"""
    
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        temp_pdf.write(empty_pdf_content)
        temp_pdf.flush()
        files = {"file": ("empty.pdf", open(temp_pdf.name, "rb"), "application/pdf")}
        response = make_request('POST', f"{API_BASE_URL}/process-pdf", files=files)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "No text extracted from PDF" in data["error"]

if __name__ == "__main__":
    logger.info("Starting API tests")
    pytest.main([__file__, "-v"]) 