#!/usr/bin/env python
# coding: utf-8
#
"""
PDF Document Retrieval API (api.py)

This file implements a comprehensive API for PDF document processing and semantic retrieval, providing the following features:
1. PDF text extraction and preprocessing
2. Text vectorization and index creation
3. Semantic search and relevant content retrieval
4. API status monitoring

This API integrates and extends the functionality of the following files:
- pdf_convert.py: Fully incorporates PDF text extraction, preprocessing, and vectorization capabilities
- query_process.py: Fully incorporates query processing and vector search functionality

Key improvements:
1. Encapsulates standalone script functionality as a Web API
2. Adds file upload and processing capabilities
3. Implements complete query and retrieval workflow
4. Enhances error handling and state management
5. Provides persistent storage functionality

API endpoints:
- GET /api/status: Retrieve API status information
- POST /api/process-pdf: Process PDF files and create vector indices
- POST /api/query: Return relevant content based on queries
- POST /api/reprocess-pdf: Re-process an uploaded PDF file
- GET /api/list-files: List PDF files in the uploads directory

Usage:
1. Start the API service: python api.py
2. The API will run at http://localhost:5000
3. Access API endpoints using HTTP clients (browser, Postman, curl)

Dependencies:
- Flask: Web framework
- PyMuPDF (fitz): PDF processing
- spaCy: Natural language processing
- sentence-transformers: Text vectorization
- FAISS: Vector indexing and search

Author: Borui Meng
Version: 1.0
Date: 2025-03-13
"""
from flask import Flask, request, jsonify
import os
import fitz
import spacy
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable cross-origin request support

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure model and index folder
MODEL_FOLDER = 'models'
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Global variables
sentence_transformer = None
vector_db = None
sentences = []

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ''
    reader = fitz.open(file_path)
    for page in reader:
        text += page.get_text('text') + '\n'
    return text.strip()

def preprocess_text(text, lowercase=True):
    """Preprocess text and split into sentences"""
    if lowercase:
        text = text.lower()
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = re.sub(r'http\S+|www\.\S+', '', text)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]   
    return sentences

def preprocess_query(query, lowercase=True):
    """Preprocess query text"""
    if lowercase:
        query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    query = re.sub(r'\s+', ' ', query).strip()
    doc = nlp(query)
    query_tokens = ' '.join([token.text for token in doc if not token.is_stop])
    return query_tokens

def load_or_create_model():
    """Load or create Sentence Transformer model"""
    global sentence_transformer
    model_path = os.path.join(MODEL_FOLDER, 'sentence_transformer.model')
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        sentence_transformer = SentenceTransformer(model_path)
    else:
        print("Creating new model")
        sentence_transformer = SentenceTransformer('sentence-t5-large')
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        sentence_transformer.save(model_path)
    
    return sentence_transformer

def text_to_vector(text):
    """Convert text to vector"""
    global sentence_transformer
    if sentence_transformer is None:
        load_or_create_model()
    return sentence_transformer.encode(text)

def create_vector_db(sentences_vector):
    """Create vector database"""
    vector_dimension = sentences_vector[0].shape[0]
    db = faiss.IndexFlatL2(vector_dimension)
    all_vectors = np.array(sentences_vector, dtype=np.float32)
    db.add(all_vectors)
    return db

def save_vector_db(db, filename='vector_index.bin'):
    """Save vector database"""
    path = os.path.join(MODEL_FOLDER, filename)
    faiss.write_index(db, path)
    return path

def load_vector_db(filename='vector_index.bin'):
    """Load vector database"""
    path = os.path.join(MODEL_FOLDER, filename)
    if os.path.exists(path):
        return faiss.read_index(path)
    return None

@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    """Process uploaded PDF file, extract text and create vector index"""
    global vector_db, sentences
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Extract text
            pdf_text = extract_text_from_pdf(file_path)
            
            # Preprocess text
            sentences = preprocess_text(pdf_text)
            
            # Ensure model is loaded
            if sentence_transformer is None:
                load_or_create_model()
            
            # Vectorize sentences
            sentences_vector = [text_to_vector(sentence) for sentence in sentences]
            
            # Create vector database
            vector_db = create_vector_db(sentences_vector)
            
            # Save vector database
            db_path = save_vector_db(vector_db)
            
            # Save sentences to file
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            with open(sentences_path, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'message': 'PDF processed successfully',
                'sentences_count': len(sentences),
                'vector_db_path': db_path,
                'sentences_path': sentences_path
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/query', methods=['POST'])
def query():
    """Process query request, return most relevant sentences"""
    global vector_db, sentences
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    top_k = data.get('top_k', 5)  # Default to return top 5 results
    
    try:
        # Preprocess query
        processed_query = preprocess_query(query_text)
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Vectorize query
        query_vector = text_to_vector(processed_query)
        
        # Ensure vector database is loaded
        if vector_db is None:
            vector_db = load_vector_db()
            if vector_db is None:
                return jsonify({'error': 'Vector database not found'}), 404
        
        # Ensure sentences are loaded
        if not sentences:
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            if os.path.exists(sentences_path):
                with open(sentences_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
            else:
                return jsonify({'error': 'Sentences not found'}), 404
        
        # Search for most similar sentences
        distances, indices = vector_db.search(
            np.array([query_vector], dtype=np.float32), 
            min(top_k, len(sentences))
        )
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(sentences):
                results.append({
                    'sentence': sentences[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx)
                })
        
        return jsonify({
            'query': query_text,
            'processed_query': processed_query,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Return API status information"""
    global vector_db, sentences
    
    vector_db_loaded = vector_db is not None
    model_loaded = sentence_transformer is not None
    sentences_loaded = len(sentences) > 0
    
    return jsonify({
        'status': 'running',
        'vector_db_loaded': vector_db_loaded,
        'model_loaded': model_loaded,
        'sentences_loaded': sentences_loaded,
        'sentences_count': len(sentences) if sentences_loaded else 0
    })

@app.route('/api/reprocess-pdf', methods=['POST'])
def reprocess_pdf():
    """Reprocess an already uploaded PDF file"""
    global vector_db, sentences
    
    data = request.json
    if not data or 'filename' not in data:
        # If no filename specified, try to process the first PDF file in uploads directory
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
        if not pdf_files:
            return jsonify({'error': 'No PDF files found in uploads directory'}), 404
        filename = pdf_files[0]
    else:
        filename = data['filename']
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File {filename} not found in uploads directory'}), 404
    
    try:
        print(f"Reprocessing file: {file_path}")
        
        # Extract text
        pdf_text = extract_text_from_pdf(file_path)
        print(f"Extracted text length: {len(pdf_text)}")
        
        if len(pdf_text) == 0:
            return jsonify({'error': 'No text extracted from PDF'}), 400
        
        # Preprocess text
        sentences = preprocess_text(pdf_text)
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Vectorize sentences
        sentences_vector = [text_to_vector(sentence) for sentence in sentences]
        
        # Create vector database
        vector_db = create_vector_db(sentences_vector)
        
        # Save vector database
        db_path = save_vector_db(vector_db)
        
        # Save sentences to file
        sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
        with open(sentences_path, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'message': 'PDF reprocessed successfully',
            'filename': filename,
            'sentences_count': len(sentences),
            'vector_db_path': db_path,
            'sentences_path': sentences_path
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/list-files', methods=['GET'])
def list_files():
    """List PDF files in the uploads directory"""
    pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    return jsonify({
        'files': pdf_files
    })

if __name__ == '__main__':
    # Preload model and vector database
    load_or_create_model()
    vector_db = load_vector_db()
    
    # Load sentences
    sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
    if os.path.exists(sentences_path):
        with open(sentences_path, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
    
    # Start API service
    app.run(debug=True, host='0.0.0.0', port=5000)