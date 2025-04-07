# Backend Unit Tests Report

This document summarizes the execution results of unit tests for each backend module.

---

## Test Environment
- OS: macOS
- Python version: 3.10
- Virtual environment: conda `9900proj`
- Framework: `pytest`

### Dependencies:
```bash
pip install -r tests/backend/requirements-dev.txt
```

---

## Test Case Summary

---

### 1. Module: `pdf_convert.py`
- **Functions tested**:
  - `extract_text_from_pdf`
  - `preprocess_text`
  - `text_to_vector_sentence_transformer`

**Run command:**
```bash
PYTHONPATH=. pytest tests/backend/unit/test_pdf_convert.py -s
```

**Expected result:**
```bash
tests/backend/unit/test_pdf_convert.py ...
=========================== 9 passed in XX.XXs ============================
```

**Actual result:**
- 9 tests passed

**Edge cases included:**
- Empty PDF content
- Empty text for preprocessing
- Text with no punctuation
- Special characters
- Empty input for vector encoding
- Long input for vector encoding

All tests passed successfully with the real model: "sentence-transformers/sentence-t5-large".

Screenshot available at: `tests/backend/unit/screenshots/test_pdf_convert_result.png`

---

### 2. Module: `query_process.py`
- **Functions tested**:
  - `preprocess_query`
  - `Query_processor.encode`

**Run command:**
```bash
PYTHONPATH=. pytest tests/backend/unit/test_query_process.py -s
```

**Expected result:**
```bash
tests/backend/unit/test_query_process.py .....
=========================== 5 passed in XX.XXs ============================
```

**Actual result:**
- 5 tests passed

**Edge cases included:**
- Empty query input
- Query containing only stopwords
- Empty input for vector encoding

All tests passed successfully.

Screenshot available at: `tests/backend/unit/screenshots/test_query_process_result.png`

---

### 3. Module: `retrieval.py`
- **Function tested**:
  - `retrieve_top_k_documents`

### Preparation before testing:
Before running the test below, make sure the mock vector index and sentences are generated:

```bash
PYTHONPATH=. python tests/backend/unit/generate_mock_retrieval_data.py
```

This script creates:
- `tests/backend/unit/mocks/vector_index.bin`
- `tests/backend/unit/mocks/sentences.pkl`
- `backend/models/sentence_transformer.model` (dummy file required for model path resolution)

**Run command:**
```bash
PYTHONPATH=. pytest tests/backend/unit/test_retrieval.py -s
```

**Expected result:**
```bash
tests/backend/unit/test_retrieval.py ......
=========================== 6 passed in XX.XXs ============================
```

**Actual result:**
- 6 tests passed

**Edge cases included:**
- Empty query input
- `None` as query input
- Special characters in query
- Very long query string
- `top_k` value greater than number of documents

All tests passed successfully.

Screenshot available at: `tests/backend/unit/screenshots/test_retrieval_result.png`

### 4. Module: `generation.py`
- **Function tested**:
  - `generate_answer`

**Run command:**
```bash
PYTHONPATH=. pytest tests/backend/unit/test_generation.py -s
```

**Expected result:**
```bash
tests/backend/unit/test_generation.py ...........
=========================== 11 passed in XX.XXs ============================
```

**Actual result:**
- 11 tests passed

**Edge cases included:**
- Empty query
- `None` as query
- Query with special characters
- Very long query string
- `top_k` greater than number of documents
- Missing API key
- Retrieval failure
- Empty retrieval result
- DashScope API error
- Streaming error

All tests passed successfully.

Screenshot available at: `tests/backend/unit/screenshots/test_generation_result.png`

---

## Notes
- This report only includes unit tests.
- Integration and UI test results can be documented separately.
- Screenshots are saved under `tests/backend/unit/screenshots/`.

