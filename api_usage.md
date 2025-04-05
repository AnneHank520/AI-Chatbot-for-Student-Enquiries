# PDF Document Retrieval API Documentation

This document provides comprehensive information about the available API endpoints for the PDF Document Retrieval system.

## Base URL

All API endpoints are accessible at the base URL: `http://localhost:5000/api`

## Authentication

Currently, the API does not require authentication.

## API Endpoints

### Status Check

**Endpoint**: `GET /api/status`

Retrieves the current status of the API including information about loaded models and database.

**Response Example**:
```json
{
  "status": "running",
  "vector_db_loaded": true,
  "model_loaded": true,
  "sentences_loaded": true,
  "sentences_count": 1250,
  "url_index_count": 32,
  "keywords_index_count": 568,
  "dashscope_api_available": true
}
```

### Process PDF

**Endpoint**: `POST /api/process-pdf`

Uploads and processes a PDF file, extracting text and adding it to the vector database.

**Request**: Form data with file

**Response Example**:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "sentences_count": 245,
  "processing_time": 3.45
}
```

### List Files

**Endpoint**: `GET /api/list-files`

Lists all PDF files that have been uploaded to the server.

**Response Example**:
```json
{
  "files": ["document1.pdf", "document2.pdf", "document3.pdf"]
}
```

### Reprocess PDF

**Endpoint**: `POST /api/reprocess-pdf`

Reprocesses an already uploaded PDF file to refresh the vector database entries.

**Request Body**:
```json
{
  "filename": "document.pdf",
  "remove_previous": true
}
```

**Response Example**:
```json
{
  "status": "success",
  "filename": "document.pdf",
  "sentences_count": 245,
  "processing_time": 3.45
}
```

### Delete PDF Content

**Endpoint**: `POST /api/delete-pdf-content`

Removes all content from a specific PDF file from the vector database.

**Request Body**:
```json
{
  "filename": "document.pdf"
}
```

**Response Example**:
```json
{
  "status": "success",
  "message": "Content from document.pdf has been removed from the database",
  "indices_removed": 245,
  "sentences_remaining": 1005
}
```

### Query

**Endpoint**: `POST /api/query`

Searches for relevant content based on the provided query.

**Request Body**:
```json
{
  "query": "What is machine learning?",
  "top_k": 12,
  "context_size": 5
}
```

**Parameters**:
- `query`: The search query
- `top_k`: Number of results to return (default: 12)
- `context_size`: Number of surrounding sentences to include for context (default: 5)

**Response**: Includes matching sentences, similarity scores, and a prompt template.

### Generate Answer

**Endpoint**: `POST /api/generate-answer`

Generates a comprehensive answer using an LLM based on the query and retrieved context.

**Request Body**:
```json
{
  "query": "What is machine learning?",
  "top_k": 12,
  "context_size": 5,
  "model": "qwen-plus"
}
```

**Parameters**:
- `query`: The search query
- `top_k`: Number of results to search for (default: 12)
- `context_size`: Number of surrounding sentences to include for context (default: 5)
- `model`: The model to use for answer generation (default: "qwen-plus")

**Response Example**:
```json
{
  "query": "What is machine learning?",
  "processed_query": "machine learning",
  "prompt_template": "...",
  "answer": "Machine learning is a branch of artificial intelligence that...",
  "model": "qwen-plus",
  "content_blocks_count": 8,
  "relevant_links_count": 3
}
```

### Extract URL Content

**Endpoint**: `POST /api/extract-url-content`

Extracts content from a provided URL.

**Request Body**:
```json
{
  "url": "https://example.com"
}
```

**Response Example**:
```json
{
  "status": "success",
  "title": "Example Domain",
  "content": "This domain is for use in illustrative examples...",
  "url": "https://example.com"
}
```

### Add URL to Index

**Endpoint**: `POST /api/add-url-to-index`

Extracts content from a URL and adds it to the vector database.

**Request Body**:
```json
{
  "url": "https://example.com"
}
```

**Response Example**:
```json
{
  "status": "success",
  "url": "https://example.com",
  "title": "Example Domain",
  "sentences_count": 12,
  "processing_time": 1.23
}
```

## Response Formats

The API returns responses in JSON format with HTTP status codes:
- 200: Success
- 400: Bad request (missing parameters, invalid input)
- 404: Resource not found
- 500: Server error

## Error Handling

All endpoints return error information in a consistent format:

```json
{
  "error": "Error message description"
}
```

## Recent Changes

The API has been enhanced to provide more natural and conversational responses from the large language model:

1. The output format now ensures responses are polite and conversational in tone
2. Responses begin with a brief acknowledgment of the user's question
3. Each response includes 2-3 relevant follow-up questions after the main answer
4. DashScope API key support has been integrated for enhanced API calls

## Example Usage

### JavaScript Example (Fetch API)

```javascript
// Example: Generate an answer
async function generateAnswer(query) {
  try {
    const response = await fetch('http://localhost:5000/api/generate-answer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: query,
        top_k: 12,
        context_size: 5,
        model: 'qwen-plus'
      })
    });
    
    const data = await response.json();
    
    if (data.error) {
      console.error('Error:', data.error);
      return null;
    }
    
    return data.answer;
  } catch (error) {
    console.error('Error generating answer:', error);
    return null;
  }
}
``` 