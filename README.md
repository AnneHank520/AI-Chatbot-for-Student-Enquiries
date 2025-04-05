[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=18242497&assignment_repo_type=AssignmentRepo)
# Rekro Document Management System

A comprehensive document management system with PDF processing, URL extraction, and vector database capabilities.

## Project Structure

- `frontend/`: Contains the user interface and admin interface
  - `roro-admin/`: Admin dashboard for managing PDFs and URLs
  - `roro/`: User interface for document search and interaction
- `backend/`: Contains API services and data processing modules
  - `api/`: RESTful API endpoints (All backend functionality has been integrated here)
  - `retrieval/`: Vector database and search functionality (Legacy code - not in use)
  - `data_processing/`: PDF and URL content processing (Legacy code - not in use)


## Environment Setup

### Backend Environment Setup (Python)

#### Using Conda (Recommended)

1. Install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create and activate the environment from the provided file:

```bash
# Create a new environment using environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate 9900proj
```

3. Verify your installation:

```bash
# Check Python version
python --version  # Should display Python 3.10.x

# Check if key packages are installed
pip list | grep flask
pip list | grep torch
pip list | grep spacy
pip list | grep sentence-transformers
pip list | grep faiss
```

#### Using Pip

If you prefer not to use Conda, you can use pip:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required spaCy model
python -m spacy download en_core_web_sm
```

### Frontend Environment Setup (Node.js)

1. Install [Node.js](https://nodejs.org/) (version 18 or higher)
2. Install dependencies for the admin dashboard:

```bash
cd ./frontend/roro-admin
npm install
```

3. Install dependencies for the user interface:

```bash
cd ./frontend/roro
npm install
```

## Running the Application

### Backend API Service

1. First, ensure you have created the necessary directories:

```bash
mkdir -p uploads models
```

2. Start the API service:

```bash
cd ./backend/api
python api.py
```

The API will run on `http://localhost:5000` by default.

### Frontend - Admin Dashboard

1. Configure the development server port in `frontend/roro-admin/package.json` (this should already be set to 3001):

```json
"scripts": {
  "dev": "next dev -p 3001",
  "start": "next start -p 3001"
}
```

2. Run the development server:

```bash
cd ./frontend/roro-admin
npm run dev
```

The admin dashboard will be accessible at `http://localhost:3001`.

### Frontend - User Interface

1. Run the development server (this should run on port 3000 by default):

```bash
cd ./frontend/roro
npm run dev
```

The user interface will be accessible at `http://localhost:3000`.

## Core Features

### Admin Dashboard
- System status monitoring
- PDF document management (upload, process, delete)
- URL content extraction and indexing

### User Interface
- Document search and retrieval
- Content viewing and interaction

## API Documentation

The system provides the following key API endpoints:

- `GET /api/status`: Check system status and database information
- `POST /api/process-pdf`: Upload and process PDF files
- `GET /api/list-files`: List available PDF files
- `POST /api/reprocess-pdf`: Reprocess existing PDF files
- `POST /api/delete-pdf-content`: Delete PDF content from database
- `POST /api/query`: Search PDF content
- `POST /api/extract-url-content`: Extract URL content
- `POST /api/add-url-to-index`: Add URL content to database

For more details, see the documentation in `backend/api/api_usage.md`.

## Frontend Dependencies

The frontend applications are built with:

### Core Dependencies
- Next.js: ^15.2.1 - React framework
- React: ^18.2.0 - UI library
- TypeScript: ^5.4.5 - Type checking

### UI Components
- Tailwind CSS: ^3.4.1 - Utility-first CSS framework
- Heroicons: ^2.1.1 - SVG icons

### Data Management
- Tanstack Query (React Query): ^5.28.0 - Data fetching and state management
- Axios: ^1.7.1 - HTTP client

## Troubleshooting

### Backend Issues

#### Package Conflicts
If you encounter package conflicts during installation:

```bash
# Try creating the environment with the --no-deps option
conda env create -f environment.yml --no-deps

# Then manually install key dependencies
conda activate 9900proj
pip install flask flask-cors werkzeug numpy pymupdf spacy sentence-transformers faiss-cpu torch
python -m spacy download en_core_web_sm
```

#### GPU Support
If you have an NVIDIA GPU and want to use GPU acceleration:

```bash
# After activating the environment
conda activate 9900proj

# Uninstall CPU versions
pip uninstall -y torch torchvision torchaudio faiss-cpu

# Install GPU versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

### Frontend Issues

#### Node Modules Not Found
If you get errors about missing modules:

```bash
# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

#### Port Already in Use
If port 3000 or 3001 is already in use, modify the port in the start script in package.json:

```json
"scripts": {
  "dev": "next dev -p [new_port]"
}
```

## Operating System Considerations

- **Windows**: Make sure Visual C++ Redistributable is installed
- **Linux**: You may need to install additional system libraries (such as libgl1-mesa-glx)
- **macOS**: Some packages may require additional dependencies installed via Homebrew

## Environment Export (for replicating the environment on other hosts)

If you need to export the environment to a new `environment.yml` file:

```bash
# Activate the environment
conda activate 9900proj

# Export the environment (without platform-specific build information)
conda env export --no-builds > environment.yml
```

## Alternative Installation Methods

If you don't want to use the `environment.yml` file, you can also:

1. Use the `setup_environment.py` script to automatically install all dependencies
2. Use `requirements.txt` to install dependencies via pip

Please refer to the main project README file for details.