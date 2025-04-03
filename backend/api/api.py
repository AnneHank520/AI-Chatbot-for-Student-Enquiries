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
import dashscope
from dotenv import load_dotenv
from openai import OpenAI

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
    # text = re.sub(r'http\S+|www\.\S+', '', text)  # 不再删除URL链接

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
    """Process uploaded PDF file, extract text and create vector index"""
    global vector_db, sentences, url_index, keywords_index
    
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
            
            # Convert to enriched format with URLs and keywords
            enriched_sentences = []
            for idx, sentence in enumerate(sentences):
                urls = extract_urls(sentence)
                keywords = extract_keywords(sentence)
                
                enriched_sentences.append({
                    'text': sentence,
                    'index': idx,
                    'length': len(sentence.split()),
                    'has_url': len(urls) > 0,
                    'urls': urls,
                    'keywords': keywords
                })
            
            # Ensure model is loaded
            if sentence_transformer is None:
                load_or_create_model()
            
            # Vectorize sentences
            sentences_vector = [text_to_vector(sentence['text']) for sentence in enriched_sentences]
            
            # Create vector database
            vector_db = create_vector_db(sentences_vector)
            
            # Build URL and keyword indices
            build_indices(enriched_sentences)
            
            # Save vector database
            db_path = save_vector_db(vector_db)
            
            # Save sentences to file
            sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
            with open(sentences_path, 'w', encoding='utf-8') as f:
                json.dump(enriched_sentences, f, ensure_ascii=False, indent=2)
            
            # Save URL index
            url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
            with open(url_index_path, 'w', encoding='utf-8') as f:
                json.dump(url_index, f, ensure_ascii=False, indent=2)
                
            # Save keywords index
            keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
            with open(keywords_index_path, 'w', encoding='utf-8') as f:
                json.dump(keywords_index, f, ensure_ascii=False, indent=2)
            
            sentences = enriched_sentences
            
            return jsonify({
                'message': 'PDF processed successfully',
                'sentences_count': len(sentences),
                'vector_db_path': db_path,
                'sentences_path': sentences_path,
                'url_count': len(url_index),
                'keywords_count': len(keywords_index)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
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
    """Reprocess an already uploaded PDF file"""
    global vector_db, sentences, url_index, keywords_index
    
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
        
        # Convert to enriched format with URLs and keywords
        enriched_sentences = []
        for idx, sentence in enumerate(sentences):
            urls = extract_urls(sentence)
            keywords = extract_keywords(sentence)
            
            enriched_sentences.append({
                'text': sentence,
                'index': idx,
                'length': len(sentence.split()),
                'has_url': len(urls) > 0,
                'urls': urls,
                'keywords': keywords
            })
        
        # Ensure model is loaded
        if sentence_transformer is None:
            load_or_create_model()
        
        # Vectorize sentences
        sentences_vector = [text_to_vector(sentence['text']) for sentence in enriched_sentences]
        
        # Create vector database
        vector_db = create_vector_db(sentences_vector)
        
        # Build URL and keyword indices
        build_indices(enriched_sentences)
        
        # Save vector database
        db_path = save_vector_db(vector_db)
        
        # Save sentences to file
        sentences_path = os.path.join(MODEL_FOLDER, 'sentences.json')
        with open(sentences_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_sentences, f, ensure_ascii=False, indent=2)
        
        # Save URL index
        url_index_path = os.path.join(MODEL_FOLDER, 'url_index.json')
        with open(url_index_path, 'w', encoding='utf-8') as f:
            json.dump(url_index, f, ensure_ascii=False, indent=2)
            
        # Save keywords index
        keywords_index_path = os.path.join(MODEL_FOLDER, 'keywords_index.json')
        with open(keywords_index_path, 'w', encoding='utf-8') as f:
            json.dump(keywords_index, f, ensure_ascii=False, indent=2)
        
        sentences = enriched_sentences
        
        return jsonify({
            'message': 'PDF reprocessed successfully',
            'filename': filename,
            'sentences_count': len(sentences),
            'vector_db_path': db_path,
            'sentences_path': sentences_path,
            'url_count': len(url_index),
            'keywords_count': len(keywords_index)
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