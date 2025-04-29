# Backend API End-to-End Test Report

This document summarizes the end-to-end tests conducted on the backend API for the capstone project. The tests verify core endpoints for PDF processing, querying, answer generation, and error handling.

## Test Environment

- Python: 3.10.16
- OS: macOS (Apple Silicon)
- Test Framework: `pytest`
- API Base URL: `http://localhost:5000/api`

### Dependencies:
```bash
pip install -r tests/backend/requirements-dev.txt
```


## Test Results

| Test Case                          | Description                                                | Status  |
|-----------------------------------|------------------------------------------------------------|---------|
| `test_status_endpoint`            | Verifies `/api/status` returns proper system status        | Passed |
| `test_query_endpoint`             | Validates `/api/query` with processed PDF                  | Passed |
| `test_generate_answer_endpoint`   | Tests `/api/generate-answer` with model query              | Passed |
| `test_list_files_endpoint`        | Confirms file list after processing                        | Passed |
| `test_delete_pdf_content_endpoint`| Checks that PDF content can be deleted properly            | Passed |
| `test_error_handling`             | Simulates missing/invalid inputs and corrupt PDF scenarios | Passed |
| `test_empty_pdf`                  | Handles blank/minimal valid PDF correctly                  | Passed |

Total: **7 tests passed**

## Test Design

- Test file: `tests/backend/e2e/test_api_requests.py`
- Uses `pytest` fixtures and temporary files
- Reads large test file `Resource.pdf` from `uploads/`(previously included during testing; after project simplification, if running tests, please manually create an `uploads/` folder under the project root and place `Resource.pdf` inside)
- Automatically logs results to both console and `test_api.log`
- Reads DashScope API key from `.env` file (using `python-dotenv`)

## Expected Result

All 7 test cases should pass successfully.

```
tests/backend/e2e/test_api_requests.py::test_status_endpoint PASSED
...
tests/backend/e2e/test_api_requests.py::test_empty_pdf PASSED
```

## Actual Result

- 7 tests passed.
- Screenshot available at: `tests/backend/e2e/test_api_requests_result.png`


## How to Run

```bash
PYTHONPATH=. pytest tests/backend/e2e/test_api_requests.py -v
```

## Notes

- Ensure `uploads/Resource.pdf` is present before testing.
- Set `DASHSCOPE_API_KEY` in a `.env` file at the project root.
- Make sure the backend Flask server is running at `localhost:5000`.

