[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=18242497&assignment_repo_type=AssignmentRepo)
# PDF Retrieval API Environment Installation Guide

This document provides detailed instructions for installing the PDF Retrieval API environment using the `environment.yml` file.

## Prerequisites

- Install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Ensure you have sufficient disk space on your system (at least 2GB)

## Installation Steps

### 1. Copy Project Files

First, copy the project files to a directory on the target host. Make sure to include the following files:
- `environment.yml` (environment configuration file)
- `api.py` and other related code files

### 2. Create and Activate Conda Environment

Open a terminal (Anaconda Prompt on Windows or terminal on Linux/macOS), then execute the following commands:

```bash
# Navigate to the project directory
cd path/to/project/directory

# Create a new environment using environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate 9900proj
```

> **Note**: The environment creation process may take several minutes, depending on your network speed and computer performance.

### 3. Verify Installation

After installation is complete, verify that the environment is correctly installed:

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

### 4. Create Necessary Directories

Ensure you create the directories required by the API:

```bash
# Create uploads and models directories in the project directory
mkdir -p uploads models
```

### 5. Run the API

Now you can start the API service:

```bash
python api.py
```

The API will run at http://localhost:5000.

## Troubleshooting

### Package Conflicts or Installation Failures

If you encounter package conflicts or installation failures when creating the environment:

```bash
# Try creating the environment with the --no-deps option
conda env create -f environment.yml --no-deps

# Then manually install key dependencies
conda activate 9900proj
pip install flask flask-cors werkzeug numpy pymupdf spacy sentence-transformers faiss-cpu torch
python -m spacy download en_core_web_sm
```

### GPU Support

If your host has an NVIDIA GPU and you want to use GPU acceleration:

```bash
# After activating the environment
conda activate 9900proj

# Uninstall CPU versions
pip uninstall -y torch torchvision torchaudio faiss-cpu

# Install GPU versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

> **Note**: Please adjust the cu118 part in the PyTorch installation URL according to your CUDA version.

### Operating System Considerations

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