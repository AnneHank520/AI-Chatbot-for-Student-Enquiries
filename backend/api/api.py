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
- POST /api/extract-url-content: Extract content from a URL

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
- requests: HTTP requests
- BeautifulSoup4: HTML parsing

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
import dashscope
from dotenv import load_dotenv
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import time

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
url_index = {}  # URL index, format: {url: [sentence_indices]}
keywords_index = {}  # Keywords index, format: {keyword: [sentence_indices]}

# Load environment variables
load_dotenv()

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
    # 检查向量数组是否为空
    if not sentences_vector:
        print(f"[{time.strftime('%H:%M:%S')}] Warning: Empty sentences vector array provided to create_vector_db")
        # 如果向量数组为空，使用模型的维度创建一个空的向量数据库
        if sentence_transformer is None:
            load_or_create_model()
        vector_dimension = sentence_transformer.get_sentence_embedding_dimension()
        print(f"[{time.strftime('%H:%M:%S')}] Creating empty vector database with dimension {vector_dimension}")
        db = faiss.IndexFlatL2(vector_dimension)
        return db
    
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

def extract_keywords(text, max_keywords=10):
    """Extract keywords from text"""
    doc = nlp(text)
    # Extract nouns, verbs, adjectives as keywords
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN')]
    # Sort by frequency and return top N keywords
    keyword_freq = {}
    for keyword in keywords:
        if len(keyword) > 2:  # Only keep words longer than 2 characters
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    return [k for k, v in sorted_keywords[:max_keywords]]

def extract_urls(text):
    """Extract URLs from text"""
    return re.findall(r'https?://\S+|www\.\S+', text)

def build_indices(sentences):
    """Build URL and keyword indices"""
    global url_index, keywords_index
    
    url_index = {}
    keywords_index = {}
    
    for idx, sentence in enumerate(sentences):
        # Get text content
        if isinstance(sentence, dict):
            text = sentence.get('text', '')
        else:
            text = sentence
        
        # Extract URLs
        urls = extract_urls(text)
        for url in urls:
            if url not in url_index:
                url_index[url] = []
            url_index[url].append(idx)
        
        # Extract keywords
        keywords = extract_keywords(text)
        for keyword in keywords:
            if keyword not in keywords_index:
                keywords_index[keyword] = []
            keywords_index[keyword].append(idx)
    
    print(f"URL index built: {len(url_index)} URLs")
    print(f"Keyword index built: {len(keywords_index)} keywords")

@app.route('/api/process-pdf', methods=['POST'])
def process_pdf():
    """Process uploaded PDF file, extract text and add to vector index"""
    global vector_db, sentences, url_index, keywords_index
    
    # Record start time
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Process PDF operation started")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Saving uploaded file: {filename}")
        file_save_start = time.time()
        file.save(file_path)
        file_save_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: File saved in {file_save_end - file_save_start:.2f} seconds")
        
        try:
            # Extract text
            print(f"[{time.strftime('%H:%M:%S')}] Step 2: Extracting text from PDF...")
            extract_start = time.time()
            pdf_text = extract_text_from_pdf(file_path)
            extract_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 2: Text extraction completed in {extract_end - extract_start:.2f} seconds. Extracted text length: {len(pdf_text)}")
            
            if len(pdf_text) == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No text extracted from PDF")
                return jsonify({'error': 'No text extracted from PDF'}), 400
            
            # Preprocess text
            print(f"[{time.strftime('%H:%M:%S')}] Step 3: Preprocessing text into sentences...")
            preprocess_start = time.time()
            pdf_sentences = preprocess_text(pdf_text)
            preprocess_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 3: Preprocessing completed in {preprocess_end - preprocess_start:.2f} seconds. Generated {len(pdf_sentences)} sentences")
            
            if len(pdf_sentences) == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Warning: No sentences generated from PDF")
                return jsonify({'error': 'No sentences generated from PDF text'}), 400
            
            # Ensure model is loaded
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Ensuring model is loaded...")
            if sentence_transformer is None:
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Loading model...")
                model_start = time.time()
                load_or_create_model()
                model_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Model loaded in {model_end - model_start:.2f} seconds")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 4: Model already loaded")
            
            # Ensure vector database and sentences are loaded
            print(f"[{time.strftime('%H:%M:%S')}] Step 5: Loading vector database...")
            if vector_db is None:
                db_load_start = time.time()
                vector_db = load_vector_db()
                if vector_db is None:
                    # If no existing database, create a new one
                    print(f"[{time.strftime('%H:%M:%S')}] Step 5: No existing database found, creating a new one")
                    vector_db = faiss.IndexFlatL2(sentence_transformer.get_sentence_embedding_dimension())
                db_load_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 5: Vector database loaded/created in {db_load_end - db_load_start:.2f} seconds")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 5: Vector database already loaded")
            
            print(f"[{time.strftime('%H:%M:%S')}] Step 6: Loading sentences...")
            if not sentences:
                sentences_load_start = time.time()
                sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
                if os.path.exists(sentences_path):
                    with open(sentences_path, 'r', encoding='utf-8') as f:
                        sentences = json.load(f)
                    print(f"[{time.strftime('%H:%M:%S')}] Step 6: Loaded {len(sentences)} sentences from file")
                else:
                    sentences = []
                    print(f"[{time.strftime('%H:%M:%S')}] Step 6: No sentences file found, starting with empty list")
                sentences_load_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: Sentences loaded in {sentences_load_end - sentences_load_start:.2f} seconds")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: Sentences already loaded ({len(sentences)} sentences)")
            
            # Load URL and keyword indices if needed
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loading indices...")
            indices_load_start = time.time()
            if not url_index:
                url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
                if os.path.exists(url_index_path):
                    with open(url_index_path, 'r', encoding='utf-8') as f:
                        url_index = json.load(f)
                    print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loaded URL index with {len(url_index)} entries")
                else:
                    url_index = {}
                    print(f"[{time.strftime('%H:%M:%S')}] Step 7: No URL index found, starting with empty dict")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: URL index already loaded ({len(url_index)} entries)")
            
            if not keywords_index:
                keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
                if os.path.exists(keywords_index_path):
                    with open(keywords_index_path, 'r', encoding='utf-8') as f:
                        keywords_index = json.load(f)
                    print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loaded keywords index with {len(keywords_index)} entries")
                else:
                    keywords_index = {}
                    print(f"[{time.strftime('%H:%M:%S')}] Step 7: No keywords index found, starting with empty dict")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: Keywords index already loaded ({len(keywords_index)} entries)")
            indices_load_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: Indices loaded in {indices_load_end - indices_load_start:.2f} seconds")
            
            # Get current sentences count for indexing
            start_idx = len(sentences)
            
            # Convert to enriched format with URLs and keywords
            print(f"[{time.strftime('%H:%M:%S')}] Step 8: Creating enriched sentences...")
            enrich_start = time.time()
            enriched_sentences = []
            for idx, sentence in enumerate(pdf_sentences):
                full_idx = start_idx + idx
                
                # Add metadata about the PDF source
                metadata = {'source': 'pdf', 'filename': filename}
                
                urls = extract_urls(sentence)
                keywords = extract_keywords(sentence)
                
                enriched_sentence = {
                    'text': sentence,
                    'index': full_idx,
                    'length': len(sentence.split()),
                    'has_url': len(urls) > 0,
                    'urls': urls,
                    'keywords': keywords,
                    'metadata': metadata
                }
                
                enriched_sentences.append(enriched_sentence)
                sentences.append(enriched_sentence)  # Add to global sentences
                
                if (idx + 1) % 1000 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Step 8: Processed {idx + 1}/{len(pdf_sentences)} sentences")
            
            enrich_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 8: Enrichment completed in {enrich_end - enrich_start:.2f} seconds. Added {len(enriched_sentences)} enriched sentences.")
            
            # Vectorize new sentences
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Vectorizing new sentences...")
            vectorize_start = time.time()
            
            # Add batch processing for vectorization
            sentences_vector = []
            batch_size = 100
            for i in range(0, len(enriched_sentences), batch_size):
                batch_end = min(i + batch_size, len(enriched_sentences))
                print(f"[{time.strftime('%H:%M:%S')}] Step 9: Vectorizing batch {i//batch_size + 1}/{(len(enriched_sentences)-1)//batch_size + 1} ({i}-{batch_end})")
                batch = [s['text'] for s in enriched_sentences[i:batch_end]]
                batch_vectors = [text_to_vector(text) for text in batch]
                sentences_vector.extend(batch_vectors)
            
            vectorize_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Vectorization completed in {vectorize_end - vectorize_start:.2f} seconds")
            
            # Convert to numpy array and add to database
            print(f"[{time.strftime('%H:%M:%S')}] Step 10: Adding vectors to database...")
            add_start = time.time()
            sentences_vector_np = np.array(sentences_vector, dtype=np.float32)
            vector_db.add(sentences_vector_np)
            add_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 10: Vectors added to database in {add_end - add_start:.2f} seconds")
            
            # Update indices
            print(f"[{time.strftime('%H:%M:%S')}] Step 11: Updating indices...")
            indices_update_start = time.time()
            for idx, sentence in enumerate(enriched_sentences):
                full_idx = start_idx + idx
                
                # Update URL index
                for url in sentence['urls']:
                    if url not in url_index:
                        url_index[url] = []
                    if full_idx not in url_index[url]:
                        url_index[url].append(full_idx)
                
                # Update keywords index
                for keyword in sentence['keywords']:
                    if keyword not in keywords_index:
                        keywords_index[keyword] = []
                    keywords_index[keyword].append(full_idx)
                
                if (idx + 1) % 1000 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] Step 11: Updated indices for {idx + 1}/{len(enriched_sentences)} sentences")
            
            indices_update_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 11: Indices updated in {indices_update_end - indices_update_start:.2f} seconds")
            
            # Save updated data
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Saving data to disk...")
            save_start = time.time()
            
            # Save vector database
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Saving vector database...")
            db_path = save_vector_db(vector_db)
            
            # Save sentences to file
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Saving sentences...")
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            with open(sentences_path, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            
            # Save URL index
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Saving URL index...")
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            with open(url_index_path, 'w', encoding='utf-8') as f:
                json.dump(url_index, f, ensure_ascii=False, indent=2)
                
            # Save keywords index
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Saving keywords index...")
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            with open(keywords_index_path, 'w', encoding='utf-8') as f:
                json.dump(keywords_index, f, ensure_ascii=False, indent=2)
            
            save_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: All data saved in {save_end - save_start:.2f} seconds")
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"[{time.strftime('%H:%M:%S')}] Process PDF operation completed in {total_time:.2f} seconds")
            
            return jsonify({
                'message': 'PDF processed successfully and added to existing index',
                'filename': filename,
                'sentences_added': len(enriched_sentences),
                'total_sentences': len(sentences),
                'vector_db_path': db_path,
                'url_count': len(url_index),
                'keywords_count': len(keywords_index),
                'processing_time_seconds': total_time
            })
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: Process PDF operation failed after {total_time:.2f} seconds")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e), 'processing_time_seconds': total_time}), 500
    
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/api/query', methods=['POST'])
def query():
    """Process query request, return most relevant sentences"""
    global vector_db, sentences, url_index, keywords_index
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    top_k = data.get('top_k', 12)  # Default to return top 12 results
    context_size = data.get('context_size', 5)  # Get context size parameter, default to 5
    
    try:
        # Preprocess query
        processed_query = preprocess_query(query_text)
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Load URL and keyword indices if needed
        if not url_index:
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            if os.path.exists(url_index_path):
                with open(url_index_path, 'r', encoding='utf-8') as f:
                    url_index = json.load(f)
        
        if not keywords_index:
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            if os.path.exists(keywords_index_path):
                with open(keywords_index_path, 'r', encoding='utf-8') as f:
                    keywords_index = json.load(f)
        
        # Extract keywords from query
        query_keywords = extract_keywords(query_text)
        
        # Find potentially relevant sentences based on keywords
        keyword_relevant_indices = set()
        for keyword in query_keywords:
            if keyword in keywords_index:
                keyword_relevant_indices.update(keywords_index[keyword])
        
        # Extract URLs from query
        query_urls = extract_urls(query_text)
        
        # Find sentences containing the URLs
        url_relevant_indices = set()
        for url in query_urls:
            if url in url_index:
                url_relevant_indices.update(url_index[url])
        
        # Combine keyword and URL matches
        boost_indices = keyword_relevant_indices.union(url_relevant_indices)
        
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
            min(top_k * 2, len(sentences))  # Retrieve more results for re-ranking
        )
        
        # Prepare final results, prioritizing keyword or URL matches
        final_indices = []
        for idx in indices[0]:
            if idx in boost_indices:
                final_indices.insert(0, int(idx))  # Prioritize matches
            else:
                final_indices.append(int(idx))
        
        # Remove duplicates and limit to top_k
        final_indices = list(dict.fromkeys(final_indices))[:top_k]
        
        # Extract URLs and organize results
        results = []
        content_blocks = []
        relevant_links = []
        processed_indices = set()  # For tracking processed indices
        
        # Process search results
        for idx in final_indices:
            if idx < len(sentences):
                # Skip if already processed
                if idx in processed_indices:
                    continue
                
                # Get current sentence
                sentence_obj = sentences[idx]
                
                # Handle different data formats
                if isinstance(sentence_obj, dict):
                    sentence_text = sentence_obj.get('text', '')
                    current_index = sentence_obj.get('index', idx)
                    # Get URLs directly from sentence object if available
                    urls = sentence_obj.get('urls', extract_urls(sentence_text))
                else:
                    sentence_text = sentence_obj
                    current_index = idx
                    urls = extract_urls(sentence_text)
                
                # Merge context - get surrounding sentences
                context_text = sentence_text
                context_indices = [int(current_index)]  # Convert to native Python int
                
                # Add preceding context
                for j in range(1, context_size + 1):
                    prev_idx = current_index - j
                    if prev_idx >= 0 and prev_idx < len(sentences):
                        prev_obj = sentences[prev_idx]
                        if isinstance(prev_obj, dict):
                            prev_text = prev_obj.get('text', '')
                            prev_urls = prev_obj.get('urls', extract_urls(prev_text))
                            urls.extend(prev_urls)
                        else:
                            prev_text = prev_obj
                            prev_urls = extract_urls(prev_text)
                            urls.extend(prev_urls)
                        context_text = prev_text + " " + context_text
                        context_indices.append(int(prev_idx))  # Convert to native Python int
                        processed_indices.add(prev_idx)
                
                # Add following context
                for j in range(1, context_size + 1):
                    next_idx = current_index + j
                    if next_idx < len(sentences):
                        next_obj = sentences[next_idx]
                        if isinstance(next_obj, dict):
                            next_text = next_obj.get('text', '')
                            next_urls = next_obj.get('urls', extract_urls(next_text))
                            urls.extend(next_urls)
                        else:
                            next_text = next_obj
                            next_urls = extract_urls(next_text)
                            urls.extend(next_urls)
                        context_text = context_text + " " + next_text
                        context_indices.append(int(next_idx))  # Convert to native Python int
                        processed_indices.add(next_idx)
                
                # Calculate document similarity to query
                try:
                    distance = float(distances[0][list(indices[0]).index(idx)])
                except ValueError:
                    # If it's a keyword/URL recommended sentence, might not be in original indices
                    doc_vector = text_to_vector(sentence_text)
                    distance = np.linalg.norm(query_vector - doc_vector)
                
                # Add to detailed results list
                results.append({
                    'sentence': context_text,
                    'distance': distance,
                    'indices': context_indices,
                    'urls': list(set(urls)),  # Deduplicate
                    'has_keyword_match': idx in keyword_relevant_indices,
                    'has_url_match': idx in url_relevant_indices
                })
                
                # Add to content blocks
                content_blocks.append(context_text)
                
                # Extract URLs
                for url in urls:
                    if url not in relevant_links:
                        relevant_links.append(url)
                
                # Mark current index as processed
                processed_indices.add(idx)
        
        # Create prompt template
        prompt_template = generate_prompt_template(query_text, content_blocks, relevant_links)
        
        # Build response structure
        response = {
            'query': query_text,
            'processed_query': processed_query,
            'keywords': query_keywords,
            'results': results,
            'prompt_data': {
                'text_content': content_blocks,
                'relevant_links': relevant_links,
                'prompt_template': prompt_template
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_prompt_template(query, content_blocks, relevant_links):
    """Generate formatted prompt template"""
    
    # Split content into numbered items
    numbered_content = []
    for i, block in enumerate(content_blocks):
        numbered_content.append(f"{i+1}. {block}")
    
    # Format links
    formatted_links = []
    for i, link in enumerate(relevant_links):
        formatted_links.append(f"{i+1}. [{link}]({link})")
    
    # Build prompt template
    template = f"""# System Instructions
You are an intelligent information assistant that answers user questions based on retrieved information and your own knowledge base. Please follow these principles:
1. Prioritize using retrieved information from reliable document sources to answer questions
2. When retrieved information is insufficient, you may supplement with your own knowledge, but clearly distinguish between retrieved content and your knowledge contributions
3. Consider retrieved content as highly reliable information sources, with priority over your own knowledge
4. If retrieved results conflict with your knowledge, defer to the retrieved results
5. Reference URLs at appropriate points in your response when relevant
6. Use concise, clear language and maintain readability in your responses
7. For region-specific information (such as Australian regulations, services, etc.), consider retrieved information as more accurate sources

# Output Format Instructions
Your response must be polite, professional, and conversational in tone. Begin your answer with a brief acknowledgment of the question in a polite way. After providing a comprehensive answer, include 2-3 relevant follow-up questions that might help the user explore the topic further.

# User Question
{query}

# Retrieved Information
## Text Content
{chr(10).join(numbered_content)}

## Relevant Links
{chr(10).join(formatted_links if formatted_links else ['No relevant links found'])}

Please answer the user's question based on the retrieved information and your knowledge base. Your answer should be direct, comprehensive, and accurate. Pay particular attention to specific details and data provided in the retrieved text, while using your knowledge to organize and supplement the information appropriately. If referencing URLs, please use markdown format for links.
"""
    
    return template

@app.route('/api/status', methods=['GET'])
def status():
    """Return API status information"""
    global vector_db, sentences, url_index, keywords_index
    
    vector_db_loaded = vector_db is not None
    model_loaded = sentence_transformer is not None
    sentences_loaded = len(sentences) > 0
    
    # Check if dashscope API key is set
    dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')
    dashscope_available = dashscope_api_key is not None and len(dashscope_api_key) > 0
    
    return jsonify({
        'status': 'running',
        'vector_db_loaded': vector_db_loaded,
        'model_loaded': model_loaded,
        'sentences_loaded': sentences_loaded,
        'sentences_count': len(sentences) if sentences_loaded else 0,
        'url_index_count': len(url_index) if url_index else 0,
        'keywords_index_count': len(keywords_index) if keywords_index else 0,
        'dashscope_api_available': dashscope_available
    })

@app.route('/api/reprocess-pdf', methods=['POST'])
def reprocess_pdf():
    """Reprocess an already uploaded PDF file and add to vector database"""
    global vector_db, sentences, url_index, keywords_index
    
    # Record start time
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Reprocess operation started")
    
    data = request.json
    if not data or 'filename' not in data:
        # If no filename specified, try to process the first PDF file in uploads directory
        pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
        if not pdf_files:
            return jsonify({'error': 'No PDF files found in uploads directory'}), 404
        filename = pdf_files[0]
    else:
        filename = data['filename']
    
    print(f"[{time.strftime('%H:%M:%S')}] Processing file: {filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': f'File {filename} not found in uploads directory'}), 404
    
    # Check if we should remove previous entries for this file - default is True
    remove_previous = data.get('remove_previous', True)
    print(f"[{time.strftime('%H:%M:%S')}] Remove previous entries: {remove_previous}")
    
    try:
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Starting reprocessing file: {file_path}")
        
        # Extract text
        print(f"[{time.strftime('%H:%M:%S')}] Step 2: Extracting text from PDF...")
        extract_start = time.time()
        pdf_text = extract_text_from_pdf(file_path)
        extract_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2: Text extraction completed in {extract_end - extract_start:.2f} seconds. Extracted text length: {len(pdf_text)}")
        
        if len(pdf_text) == 0:
            return jsonify({'error': 'No text extracted from PDF'}), 400
        
        # Preprocess text
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Preprocessing text into sentences...")
        preprocess_start = time.time()
        pdf_sentences = preprocess_text(pdf_text)
        preprocess_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Preprocessing completed in {preprocess_end - preprocess_start:.2f} seconds. Generated {len(pdf_sentences)} sentences")
        
        # Ensure model is loaded
        print(f"[{time.strftime('%H:%M:%S')}] Step 4: Ensuring model is loaded...")
        if sentence_transformer is None:
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Loading model...")
            model_start = time.time()
            load_or_create_model()
            model_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Model loaded in {model_end - model_start:.2f} seconds")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 4: Model already loaded")
        
        # Ensure vector database and sentences are loaded
        print(f"[{time.strftime('%H:%M:%S')}] Step 5: Loading vector database...")
        if vector_db is None:
            db_load_start = time.time()
            vector_db = load_vector_db()
            if vector_db is None:
                # If no existing database, create a new one
                print(f"[{time.strftime('%H:%M:%S')}] Step 5: No existing database found, creating a new one")
                vector_db = faiss.IndexFlatL2(sentence_transformer.get_sentence_embedding_dimension())
            db_load_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 5: Vector database loaded/created in {db_load_end - db_load_start:.2f} seconds")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 5: Vector database already loaded")
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 6: Loading sentences...")
        if not sentences:
            sentences_load_start = time.time()
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            if os.path.exists(sentences_path):
                with open(sentences_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: Loaded {len(sentences)} sentences from file")
            else:
                sentences = []
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: No sentences file found, starting with empty list")
            sentences_load_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 6: Sentences loaded in {sentences_load_end - sentences_load_start:.2f} seconds")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 6: Sentences already loaded ({len(sentences)} sentences)")
        
        # Load URL and keyword indices if needed
        print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loading indices...")
        indices_load_start = time.time()
        if not url_index:
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            if os.path.exists(url_index_path):
                with open(url_index_path, 'r', encoding='utf-8') as f:
                    url_index = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loaded URL index with {len(url_index)} entries")
            else:
                url_index = {}
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: No URL index found, starting with empty dict")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: URL index already loaded ({len(url_index)} entries)")
        
        if not keywords_index:
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            if os.path.exists(keywords_index_path):
                with open(keywords_index_path, 'r', encoding='utf-8') as f:
                    keywords_index = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: Loaded keywords index with {len(keywords_index)} entries")
            else:
                keywords_index = {}
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: No keywords index found, starting with empty dict")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: Keywords index already loaded ({len(keywords_index)} entries)")
        indices_load_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 7: Indices loaded in {indices_load_end - indices_load_start:.2f} seconds")
        
        # If requested, remove previous entries for this file
        print(f"[{time.strftime('%H:%M:%S')}] Step 8: Finding entries to remove...")
        indices_to_remove = []
        if remove_previous:
            # Find all sentences from this file
            find_indices_start = time.time()
            for idx, sentence in enumerate(sentences):
                if isinstance(sentence, dict) and sentence.get('metadata', {}).get('source') == 'pdf' and sentence.get('metadata', {}).get('filename') == filename:
                    indices_to_remove.append(idx)
            find_indices_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 8: Found {len(indices_to_remove)} entries to remove in {find_indices_end - find_indices_start:.2f} seconds")
            
            # Remove from url_index and keywords_index
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Updating URL and keywords indices...")
            indices_update_start = time.time()
            for url, indices in list(url_index.items()):
                url_index[url] = [idx for idx in indices if idx not in indices_to_remove]
                if not url_index[url]:
                    del url_index[url]
            
            for keyword, indices in list(keywords_index.items()):
                keywords_index[keyword] = [idx for idx in indices if idx not in indices_to_remove]
                if not keywords_index[keyword]:
                    del keywords_index[keyword]
            indices_update_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Indices updated in {indices_update_end - indices_update_start:.2f} seconds")
            
            # Note: We can't easily remove vectors from FAISS index, so we'll rebuild the index
            # This is a workaround - in production you might want to use a different approach
            if indices_to_remove:
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Rebuilding sentences list and vector database...")
                rebuild_start = time.time()
                
                # Get all sentences that are not from this file
                updated_sentences = [s for i, s in enumerate(sentences) if i not in indices_to_remove]
                sentences = updated_sentences
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Updated sentences list, now contains {len(sentences)} entries")
                
                # Rebuild FAISS index
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Vectorizing {len(sentences)} sentences...")
                vectorize_start = time.time()
                
                # Add batch processing for vectorization
                sentences_vector = []
                batch_size = 100
                for i in range(0, len(sentences), batch_size):
                    batch_end = min(i + batch_size, len(sentences))
                    print(f"[{time.strftime('%H:%M:%S')}] Step 10: Vectorizing batch {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1} ({i}-{batch_end})")
                    batch = [s['text'] for s in sentences[i:batch_end]]
                    batch_vectors = [text_to_vector(text) for text in batch]
                    sentences_vector.extend(batch_vectors)
                
                vectorize_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Vectorization completed in {vectorize_end - vectorize_start:.2f} seconds")
                
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Creating vector database...")
                if sentences_vector:
                    db_create_start = time.time()
                    vector_db = create_vector_db(sentences_vector)
                    db_create_end = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] Step 10: Vector database created in {db_create_end - db_create_start:.2f} seconds")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Step 10: No sentences to vectorize, creating empty database")
                    vector_db = faiss.IndexFlatL2(sentence_transformer.get_sentence_embedding_dimension())
                
                rebuild_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 10: Database rebuilding completed in {rebuild_end - rebuild_start:.2f} seconds")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 8-10: Skipping removal of previous entries")
        
        # Get current sentences count for indexing
        start_idx = len(sentences)
        
        # Convert to enriched format with URLs and keywords
        print(f"[{time.strftime('%H:%M:%S')}] Step 11: Creating enriched sentences...")
        enrich_start = time.time()
        enriched_sentences = []
        for idx, sentence in enumerate(pdf_sentences):
            full_idx = start_idx + idx
            
            # Add metadata about the PDF source
            metadata = {'source': 'pdf', 'filename': filename}
            
            urls = extract_urls(sentence)
            keywords = extract_keywords(sentence)
            
            enriched_sentence = {
                'text': sentence,
                'index': full_idx,
                'length': len(sentence.split()),
                'has_url': len(urls) > 0,
                'urls': urls,
                'keywords': keywords,
                'metadata': metadata
            }
            
            enriched_sentences.append(enriched_sentence)
            sentences.append(enriched_sentence)  # Add to global sentences
            
            if (idx + 1) % 1000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Step 11: Processed {idx + 1}/{len(pdf_sentences)} sentences")
        
        enrich_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 11: Enrichment completed in {enrich_end - enrich_start:.2f} seconds. Added {len(enriched_sentences)} enriched sentences.")
        
        # Vectorize new sentences
        print(f"[{time.strftime('%H:%M:%S')}] Step 12: Vectorizing new sentences...")
        vectorize_start = time.time()
        
        # Add batch processing for new sentence vectorization
        sentences_vector = []
        batch_size = 100
        for i in range(0, len(enriched_sentences), batch_size):
            batch_end = min(i + batch_size, len(enriched_sentences))
            print(f"[{time.strftime('%H:%M:%S')}] Step 12: Vectorizing batch {i//batch_size + 1}/{(len(enriched_sentences)-1)//batch_size + 1} ({i}-{batch_end})")
            batch = [s['text'] for s in enriched_sentences[i:batch_end]]
            batch_vectors = [text_to_vector(text) for text in batch]
            sentences_vector.extend(batch_vectors)
        
        vectorize_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 12: New sentences vectorization completed in {vectorize_end - vectorize_start:.2f} seconds")
        
        print(f"[{time.strftime('%H:%M:%S')}] Step 13: Converting to numpy array and adding to database...")
        sentences_vector_np = np.array(sentences_vector, dtype=np.float32)
        
        # Add vectors to database
        add_start = time.time()
        vector_db.add(sentences_vector_np)
        add_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 13: Added vectors to database in {add_end - add_start:.2f} seconds")
        
        # Update indices
        print(f"[{time.strftime('%H:%M:%S')}] Step 14: Updating indices...")
        indices_update_start = time.time()
        for idx, sentence in enumerate(enriched_sentences):
            full_idx = start_idx + idx
            
            # Update URL index
            for url in sentence['urls']:
                if url not in url_index:
                    url_index[url] = []
                if full_idx not in url_index[url]:
                    url_index[url].append(full_idx)
            
            # Update keywords index
            for keyword in sentence['keywords']:
                if keyword not in keywords_index:
                    keywords_index[keyword] = []
                keywords_index[keyword].append(full_idx)
                
            if (idx + 1) % 1000 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] Step 14: Updated indices for {idx + 1}/{len(enriched_sentences)} sentences")
        
        indices_update_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 14: Indices updated in {indices_update_end - indices_update_start:.2f} seconds")
        
        # Save updated data
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: Saving data to disk...")
        save_start = time.time()
        
        # Save vector database
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: Saving vector database...")
        db_path = save_vector_db(vector_db)
        
        # Save sentences to file
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: Saving sentences...")
        sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
        with open(sentences_path, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        # Save URL index
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: Saving URL index...")
        url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
        with open(url_index_path, 'w', encoding='utf-8') as f:
            json.dump(url_index, f, ensure_ascii=False, indent=2)
            
        # Save keywords index
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: Saving keywords index...")
        keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
        with open(keywords_index_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_index, f, ensure_ascii=False, indent=2)
        
        save_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 15: All data saved in {save_end - save_start:.2f} seconds")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Reprocess operation completed in {total_time:.2f} seconds")
        
        return jsonify({
            'message': 'PDF reprocessed successfully and added to index',
            'filename': filename,
            'sentences_added': len(enriched_sentences),
            'total_sentences': len(sentences),
            'previous_entries_removed': len(indices_to_remove) if remove_previous else 0,
            'vector_db_path': db_path,
            'url_count': len(url_index),
            'keywords_count': len(keywords_index),
            'processing_time_seconds': total_time
        })
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Reprocess operation failed after {total_time:.2f} seconds")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'processing_time_seconds': total_time}), 500

@app.route('/api/list-files', methods=['GET'])
def list_files():
    """List PDF files in the uploads directory"""
    pdf_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.pdf')]
    return jsonify({
        'files': pdf_files
    })

@app.route('/api/generate-answer', methods=['POST'])
def generate_answer_api():
    """Generate answer using LLM based on the query and retrieved context"""
    global vector_db, sentences, url_index, keywords_index
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query_text = data['query']
    top_k = data.get('top_k', 12)  # Default to return top 12 results
    context_size = data.get('context_size', 5)  # Get context size parameter, default to 5
    model = data.get('model', 'qwen-plus')  # Model to use, default to qwen-plus
    
    try:
        # First, query to get the relevant content and prompt template
        # This reuses the query function's logic to get relevant content
        # Preprocess query
        processed_query = preprocess_query(query_text)
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Load URL and keyword indices if needed
        if not url_index:
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            if os.path.exists(url_index_path):
                with open(url_index_path, 'r', encoding='utf-8') as f:
                    url_index = json.load(f)
        
        if not keywords_index:
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            if os.path.exists(keywords_index_path):
                with open(keywords_index_path, 'r', encoding='utf-8') as f:
                    keywords_index = json.load(f)
        
        # Extract keywords from query
        query_keywords = extract_keywords(query_text)
        
        # Find potentially relevant sentences based on keywords
        keyword_relevant_indices = set()
        for keyword in query_keywords:
            if keyword in keywords_index:
                keyword_relevant_indices.update(keywords_index[keyword])
        
        # Extract URLs from query
        query_urls = extract_urls(query_text)
        
        # Find sentences containing the URLs
        url_relevant_indices = set()
        for url in query_urls:
            if url in url_index:
                url_relevant_indices.update(url_index[url])
        
        # Combine keyword and URL matches
        boost_indices = keyword_relevant_indices.union(url_relevant_indices)
        
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
            min(top_k * 2, len(sentences))  # Retrieve more results for re-ranking
        )
        
        # Prepare final results, prioritizing keyword or URL matches
        final_indices = []
        for idx in indices[0]:
            if idx in boost_indices:
                final_indices.insert(0, int(idx))  # Prioritize matches
            else:
                final_indices.append(int(idx))
        
        # Remove duplicates and limit to top_k
        final_indices = list(dict.fromkeys(final_indices))[:top_k]
        
        # Extract relevant content
        content_blocks = []
        relevant_links = []
        processed_indices = set()
        
        # Process search results
        for idx in final_indices:
            if idx < len(sentences):
                # Skip if already processed
                if idx in processed_indices:
                    continue
                
                # Get current sentence
                sentence_obj = sentences[idx]
                
                # Handle different data formats
                if isinstance(sentence_obj, dict):
                    sentence_text = sentence_obj.get('text', '')
                    current_index = sentence_obj.get('index', idx)
                    # Get URLs directly from sentence object if available
                    urls = sentence_obj.get('urls', extract_urls(sentence_text))
                else:
                    sentence_text = sentence_obj
                    current_index = idx
                    urls = extract_urls(sentence_text)
                
                # Merge context - get surrounding sentences
                context_text = sentence_text
                context_indices = [int(current_index)]
                
                # Add preceding context
                for j in range(1, context_size + 1):
                    prev_idx = current_index - j
                    if prev_idx >= 0 and prev_idx < len(sentences):
                        prev_obj = sentences[prev_idx]
                        if isinstance(prev_obj, dict):
                            prev_text = prev_obj.get('text', '')
                            prev_urls = prev_obj.get('urls', extract_urls(prev_text))
                            urls.extend(prev_urls)
                        else:
                            prev_text = prev_obj
                            prev_urls = extract_urls(prev_text)
                            urls.extend(prev_urls)
                        context_text = prev_text + " " + context_text
                        context_indices.append(int(prev_idx))
                        processed_indices.add(prev_idx)
                
                # Add following context
                for j in range(1, context_size + 1):
                    next_idx = current_index + j
                    if next_idx < len(sentences):
                        next_obj = sentences[next_idx]
                        if isinstance(next_obj, dict):
                            next_text = next_obj.get('text', '')
                            next_urls = next_obj.get('urls', extract_urls(next_text))
                            urls.extend(next_urls)
                        else:
                            next_text = next_obj
                            next_urls = extract_urls(next_text)
                            urls.extend(next_urls)
                        context_text = context_text + " " + next_text
                        context_indices.append(int(next_idx))
                        processed_indices.add(next_idx)
                
                # Add to content blocks
                content_blocks.append(context_text)
                
                # Extract URLs
                for url in urls:
                    if url not in relevant_links:
                        relevant_links.append(url)
                
                # Mark current index as processed
                processed_indices.add(idx)
        
        # Create prompt template
        prompt_template = generate_prompt_template(query_text, content_blocks, relevant_links)
        
        # Call LLM API to generate the answer
        print(f"Calling model: {model}")
        
        # Check if API key is available
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            return jsonify({'error': 'DASHSCOPE_API_KEY environment variable not set'}), 500
        
        try:
            # Use OpenAI compatible mode to initialize the client
            client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            
            # Create chat completion request
            completion = client.chat.completions.create(
                model=model,  # Use the model name passed in
                messages=[
                    {'role': 'system', 'content': 'You are a helpful assistant. Be polite and professional in your responses. Begin with a brief acknowledgment of the question, provide a comprehensive answer, and end with 2-3 relevant follow-up questions to help the user explore the topic further.'},
                    {'role': 'user', 'content': prompt_template}
                ]
            )
            
            # Extract generated content from the response
            answer = completion.choices[0].message.content
            
        except Exception as e:
            error_message = f"Error calling DashScope API: {str(e)}"
            print(error_message)
            return jsonify({
                'error': error_message,
                'message': "Please check API key and model name. Refer to: https://help.aliyun.com/zh/model-studio/developer-reference/error-code"
            }), 500
        
        # Build response structure
        response_data = {
            'query': query_text,
            'processed_query': processed_query,
            'prompt_template': prompt_template,
            'answer': answer,
            'model': model,
            'content_blocks_count': len(content_blocks),
            'relevant_links_count': len(relevant_links)
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def extract_content_from_url(url):
    """
    Extract content from a given URL
    
    Args:
        url (str): The URL to extract content from
        
    Returns:
        dict: A dictionary containing:
            - status: Success or error status
            - content: Extracted text content (if successful)
            - title: Page title (if available)
            - error: Error message (if failed)
    """
    try:
        # Add user-agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make HTTP request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for 4XX/5XX status codes
        
        # Check if the content type is HTML
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            return {
                'status': 'error',
                'error': f'URL content is not HTML (Content-Type: {content_type})'
            }
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title = soup.title.text.strip() if soup.title else "No title found"
        
        # Remove script and style elements
        for script in soup(["script", "style", "header", "footer", "nav"]):
            script.extract()
            
        # Get text content
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up text: remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            'status': 'success',
            'title': title,
            'content': text,
            'url': url
        }
        
    except requests.exceptions.RequestException as e:
        return {
            'status': 'error',
            'error': f'Error fetching URL: {str(e)}',
            'url': url
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Error processing URL content: {str(e)}',
            'url': url
        }

@app.route('/api/extract-url-content', methods=['POST'])
def extract_url_content_api():
    """Extract content from a provided URL"""
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid URL format. URL must start with http:// or https://'}), 400
    
    try:
        result = extract_content_from_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/add-url-to-index', methods=['POST'])
def add_url_to_index():
    """Extract content from URL and add to vector database"""
    global vector_db, sentences, url_index, keywords_index
    
    data = request.json
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid URL format. URL must start with http:// or https://'}), 400
    
    try:
        # Extract content from URL
        result = extract_content_from_url(url)
        
        if result['status'] != 'success':
            return jsonify({
                'error': 'Failed to extract content from URL',
                'details': result.get('error', 'Unknown error')
            }), 400
        
        # Get the content and title
        content = result['content']
        title = result['title']
        
        # Preprocess the content
        url_sentences = preprocess_text(content)
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Ensure vector database and sentences are loaded
        if vector_db is None:
            vector_db = load_vector_db()
            if vector_db is None:
                return jsonify({'error': 'Vector database not found'}), 404
        
        if not sentences:
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            if os.path.exists(sentences_path):
                with open(sentences_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
            else:
                sentences = []
        
        # Get current sentences count for indexing
        start_idx = len(sentences)
        
        # Create enriched sentences
        enriched_sentences = []
        for idx, sentence in enumerate(url_sentences):
            full_idx = start_idx + idx
            # Add title metadata to first sentence
            metadata = {'source': 'url', 'url': url}
            if idx == 0:
                metadata['title'] = title
            
            # Extract keywords and URLs
            urls = extract_urls(sentence)
            keywords = extract_keywords(sentence)
            
            enriched_sentence = {
                'text': sentence,
                'index': full_idx,
                'length': len(sentence.split()),
                'has_url': len(urls) > 0 or True,  # Mark all sentences as having a URL source
                'urls': urls + [url],  # Add the source URL to all sentences
                'keywords': keywords,
                'metadata': metadata
            }
            
            enriched_sentences.append(enriched_sentence)
            sentences.append(enriched_sentence)  # Add to global sentences
        
        # Vectorize new sentences
        sentences_vector = [text_to_vector(sentence['text']) for sentence in enriched_sentences]
        sentences_vector_np = np.array(sentences_vector, dtype=np.float32)
        
        # Add vectors to database
        vector_db.add(sentences_vector_np)
        
        # Update indices
        for idx, sentence in enumerate(enriched_sentences):
            full_idx = start_idx + idx
            
            # Update URL index with the source URL
            if url not in url_index:
                url_index[url] = []
            url_index[url].append(full_idx)
            
            # Update URL index with any URLs in the text
            for text_url in sentence['urls']:
                if text_url not in url_index:
                    url_index[text_url] = []
                if full_idx not in url_index[text_url]:
                    url_index[text_url].append(full_idx)
            
            # Update keywords index
            for keyword in sentence['keywords']:
                if keyword not in keywords_index:
                    keywords_index[keyword] = []
                keywords_index[keyword].append(full_idx)
        
        # Save updated data
        # Save vector database
        db_path = save_vector_db(vector_db)
        
        # Save sentences to file
        sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
        with open(sentences_path, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        
        # Save URL index
        url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
        with open(url_index_path, 'w', encoding='utf-8') as f:
            json.dump(url_index, f, ensure_ascii=False, indent=2)
            
        # Save keywords index
        keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
        with open(keywords_index_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_index, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'message': 'URL content successfully added to index',
            'url': url,
            'title': title,
            'sentences_added': len(enriched_sentences),
            'total_sentences': len(sentences),
            'keywords_count': len(set([k for s in enriched_sentences for k in s['keywords']]))
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-pdf-content', methods=['POST'])
def delete_pdf_content():
    """Delete all content from a specific PDF file from the vector database"""
    global vector_db, sentences, url_index, keywords_index
    
    # Record start time
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] Delete PDF content operation started")
    
    data = request.json
    if not data or 'filename' not in data:
        return jsonify({'error': 'No filename provided'}), 400
    
    filename = data['filename']
    print(f"[{time.strftime('%H:%M:%S')}] Processing deletion for file: {filename}")
    
    try:
        # Load sentences if not already loaded
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Loading sentences...")
        sentences_load_start = time.time()
        if not sentences:
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            if os.path.exists(sentences_path):
                with open(sentences_path, 'r', encoding='utf-8') as f:
                    sentences = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 1: Loaded {len(sentences)} sentences from file")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 1: No sentences file found")
                return jsonify({'error': 'No sentences found in database'}), 404
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 1: Sentences already loaded ({len(sentences)} sentences)")
        sentences_load_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 1: Sentences loaded in {sentences_load_end - sentences_load_start:.2f} seconds")
        
        # Load URL and keyword indices if needed
        print(f"[{time.strftime('%H:%M:%S')}] Step 2: Loading indices...")
        indices_load_start = time.time()
        if not url_index:
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            if os.path.exists(url_index_path):
                with open(url_index_path, 'r', encoding='utf-8') as f:
                    url_index = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 2: Loaded URL index with {len(url_index)} entries")
            else:
                url_index = {}
                print(f"[{time.strftime('%H:%M:%S')}] Step 2: No URL index found, starting with empty dict")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 2: URL index already loaded ({len(url_index)} entries)")
        
        if not keywords_index:
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            if os.path.exists(keywords_index_path):
                with open(keywords_index_path, 'r', encoding='utf-8') as f:
                    keywords_index = json.load(f)
                print(f"[{time.strftime('%H:%M:%S')}] Step 2: Loaded keywords index with {len(keywords_index)} entries")
            else:
                keywords_index = {}
                print(f"[{time.strftime('%H:%M:%S')}] Step 2: No keywords index found, starting with empty dict")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Step 2: Keywords index already loaded ({len(keywords_index)} entries)")
        indices_load_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 2: Indices loaded in {indices_load_end - indices_load_start:.2f} seconds")
        
        # Find all sentences from this file
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Finding sentences to remove...")
        find_indices_start = time.time()
        indices_to_remove = []
        for idx, sentence in enumerate(sentences):
            if isinstance(sentence, dict) and sentence.get('metadata', {}).get('source') == 'pdf' and sentence.get('metadata', {}).get('filename') == filename:
                indices_to_remove.append(idx)
        find_indices_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 3: Found {len(indices_to_remove)} entries to remove in {find_indices_end - find_indices_start:.2f} seconds")
        
        if not indices_to_remove:
            print(f"[{time.strftime('%H:%M:%S')}] No content found for file {filename}")
            return jsonify({
                'message': f'No content found for file {filename}',
                'indices_removed': 0
            })
        
        # Remove from url_index and keywords_index
        print(f"[{time.strftime('%H:%M:%S')}] Step 4: Updating URL and keywords indices...")
        indices_update_start = time.time()
        urls_before = len(url_index)
        keywords_before = len(keywords_index)
        
        for url, indices in list(url_index.items()):
            url_index[url] = [idx for idx in indices if idx not in indices_to_remove]
            if not url_index[url]:
                del url_index[url]
        
        for keyword, indices in list(keywords_index.items()):
            keywords_index[keyword] = [idx for idx in indices if idx not in indices_to_remove]
            if not keywords_index[keyword]:
                del keywords_index[keyword]
        
        indices_update_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 4: Indices updated in {indices_update_end - indices_update_start:.2f} seconds")
        print(f"[{time.strftime('%H:%M:%S')}] Step 4: URL index entries: {urls_before} -> {len(url_index)}")
        print(f"[{time.strftime('%H:%M:%S')}] Step 4: Keywords index entries: {keywords_before} -> {len(keywords_index)}")
        
        # Remove sentences and rebuild vector database
        print(f"[{time.strftime('%H:%M:%S')}] Step 5: Rebuilding sentences list...")
        rebuild_start = time.time()
        updated_sentences = [s for i, s in enumerate(sentences) if i not in indices_to_remove]
        print(f"[{time.strftime('%H:%M:%S')}] Step 5: Updated sentences list, now contains {len(updated_sentences)} entries (was {len(sentences)})")
        rebuild_end = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] Step 5: Sentences list rebuilt in {rebuild_end - rebuild_start:.2f} seconds")
        
        # Check if we have sentences left
        if updated_sentences:
            sentences = updated_sentences
            
            # Ensure model is loaded
            print(f"[{time.strftime('%H:%M:%S')}] Step 6: Ensuring model is loaded...")
            if sentence_transformer is None:
                model_start = time.time()
                load_or_create_model()
                model_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: Model loaded in {model_end - model_start:.2f} seconds")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 6: Model already loaded")
                
            # Rebuild FAISS index
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: Vectorizing {len(sentences)} sentences...")
            vectorize_start = time.time()
            
            # Add batch processing for vectorization
            sentences_vector = []
            batch_size = 100
            for i in range(0, len(sentences), batch_size):
                batch_end = min(i + batch_size, len(sentences))
                print(f"[{time.strftime('%H:%M:%S')}] Step 7: Vectorizing batch {i//batch_size + 1}/{(len(sentences)-1)//batch_size + 1} ({i}-{batch_end})")
                batch = [s['text'] for s in sentences[i:batch_end]]
                batch_vectors = [text_to_vector(text) for text in batch]
                sentences_vector.extend(batch_vectors)
            
            vectorize_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 7: Vectorization completed in {vectorize_end - vectorize_start:.2f} seconds")
            
            print(f"[{time.strftime('%H:%M:%S')}] Step 8: Creating vector database...")
            if sentences_vector:
                db_create_start = time.time()
                vector_db = create_vector_db(sentences_vector)
                db_create_end = time.time()
                print(f"[{time.strftime('%H:%M:%S')}] Step 8: Vector database created in {db_create_end - db_create_start:.2f} seconds")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Step 8: No sentences to vectorize, creating empty database")
                vector_db = faiss.IndexFlatL2(sentence_transformer.get_sentence_embedding_dimension())
            
            # Save all data
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Saving data to disk...")
            save_start = time.time()
            
            # Save vector database
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Saving vector database...")
            db_path = save_vector_db(vector_db)
            
            # Save sentences to file
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Saving sentences...")
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            with open(sentences_path, 'w', encoding='utf-8') as f:
                json.dump(sentences, f, ensure_ascii=False, indent=2)
            
            # Save URL index
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Saving URL index...")
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            with open(url_index_path, 'w', encoding='utf-8') as f:
                json.dump(url_index, f, ensure_ascii=False, indent=2)
                
            # Save keywords index
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Saving keywords index...")
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            with open(keywords_index_path, 'w', encoding='utf-8') as f:
                json.dump(keywords_index, f, ensure_ascii=False, indent=2)
            
            save_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: All data saved in {save_end - save_start:.2f} seconds")
        else:
            # If no sentences left, reset everything
            print(f"[{time.strftime('%H:%M:%S')}] Step 6-8: No sentences left, resetting all data structures")
            sentences = []
            vector_db = None
            url_index = {}
            keywords_index = {}
            
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: Removing index files...")
            remove_start = time.time()
            
            # Remove index files
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            if os.path.exists(sentences_path):
                os.remove(sentences_path)
                print(f"[{time.strftime('%H:%M:%S')}] Step 9: Removed sentences file")
            
            db_path = os.path.join(MODEL_FOLDER, 'vector_index.bin')
            if os.path.exists(db_path):
                os.remove(db_path)
                print(f"[{time.strftime('%H:%M:%S')}] Step 9: Removed vector database file")
                
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            if os.path.exists(url_index_path):
                os.remove(url_index_path)
                print(f"[{time.strftime('%H:%M:%S')}] Step 9: Removed URL index file")
                
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            if os.path.exists(keywords_index_path):
                os.remove(keywords_index_path)
                print(f"[{time.strftime('%H:%M:%S')}] Step 9: Removed keywords index file")
            
            remove_end = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] Step 9: All files removed in {remove_end - remove_start:.2f} seconds")
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[{time.strftime('%H:%M:%S')}] Delete PDF content operation completed in {total_time:.2f} seconds")
        
        return jsonify({
            'message': f'Content from file {filename} deleted successfully',
            'indices_removed': len(indices_to_remove),
            'sentences_remaining': len(sentences),
            'url_count': len(url_index),
            'keywords_count': len(keywords_index),
            'processing_time_seconds': total_time
        })
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[{time.strftime('%H:%M:%S')}] ERROR: Delete PDF content operation failed after {total_time:.2f} seconds")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'processing_time_seconds': total_time}), 500

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