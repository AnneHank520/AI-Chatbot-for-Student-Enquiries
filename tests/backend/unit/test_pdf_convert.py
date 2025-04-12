import re
import tempfile
import os
import spacy
import numpy as np
from reportlab.pdfgen import canvas

from backend.data_processing.pdf_convert import extract_text_from_pdf, preprocess_text
from sentence_transformers import SentenceTransformer

# Inject spaCy and sentence transformer into the tested module
import backend.data_processing.pdf_convert as pdf_convert
nlp = spacy.load("en_core_web_sm")
pdf_convert.nlp = nlp
model = SentenceTransformer("sentence-transformers/sentence-t5-large")
pdf_convert.sentence_transformer = model

# ================================
# Utilities
# ================================

def create_test_pdf(content: str) -> str:
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name)
    c.drawString(100, 750, content)
    c.save()
    return temp_pdf.name

# ================================
# Test: extract_text_from_pdf
# ================================

def test_extract_text_from_pdf():
    sample_text = "Hello, this is a test PDF!"
    pdf_path = create_test_pdf(sample_text)

    try:
        extracted = extract_text_from_pdf(pdf_path)
        assert sample_text in extracted
    finally:
        os.remove(pdf_path)

def test_extract_text_from_pdf_empty():
    pdf_path = create_test_pdf("")
    try:
        extracted = extract_text_from_pdf(pdf_path)
        assert extracted.strip() == ""
    finally:
        os.remove(pdf_path)

# ================================
# Test: preprocess_text
# ================================

def test_preprocess_text_basic():
    text = "This is sentence one.\nThis is sentence two.\n\nThis is a new paragraph."
    expected = ["This is sentence one.", "This is sentence two.", "This is a new paragraph."]

    result = preprocess_text(text, lowercase=False)
    assert isinstance(result, list)
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        assert re.sub(r'\s+', '', r) == re.sub(r'\s+', '', e)

def test_preprocess_text_empty():
    result = preprocess_text("", lowercase=True)
    assert isinstance(result, list)
    assert result == []

def test_preprocess_text_no_punctuation():
    text = "This is a test with no punctuation"
    result = preprocess_text(text, lowercase=True)
    assert len(result) == 1
    assert "punctuation" in result[0]

def test_preprocess_text_special_chars():
    text = "Is this a question?! What about that... And symbols #$%^&*()"
    result = preprocess_text(text, lowercase=True)
    assert isinstance(result, list)
    assert any("question" in s for s in result)

# ================================
# Test: text_to_vector_sentence_transformer
# ================================

def test_text_to_vector_sentence_transformer():
    text = "This is a test sentence."
    vector = pdf_convert.text_to_vector_sentence_transformer(text)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (768,)  # Updated from 384 to 768

def test_text_to_vector_sentence_transformer_empty():
    vector = pdf_convert.text_to_vector_sentence_transformer("")
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (768,)

def test_text_to_vector_sentence_transformer_long_input():
    long_text = "This is a sentence. " * 200
    vector = pdf_convert.text_to_vector_sentence_transformer(long_text)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (768,)
