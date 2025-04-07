import os
import re
import spacy
import numpy as np
import importlib.util
import sys
from unittest import mock
from sentence_transformers import SentenceTransformer

# Correct absolute path to model
correct_model_path = os.path.abspath("backend/models/sentence_transformer.model")
module_path = os.path.abspath("backend/retrieval/query_process.py")

# Dynamically load the module with patched path
with mock.patch("os.path.abspath", return_value=correct_model_path):
    spec = importlib.util.spec_from_file_location("query_process", module_path)
    query_process = importlib.util.module_from_spec(spec)
    sys.modules["query_process"] = query_process
    spec.loader.exec_module(query_process)

# Inject actual NLP and model
query_process.nlp = spacy.load("en_core_web_sm")
query_process.Query_processor = SentenceTransformer(correct_model_path)

# Test preprocess_query
def test_preprocess_query_basic():
    raw_query = "  What are the best universities in Australia? "
    expected_keywords = ["best", "universities", "australia"]
    result = query_process.preprocess_query(raw_query, lowercase=True)

    assert isinstance(result, str)
    for word in expected_keywords:
        assert word in result
    assert "?" not in result
    assert "what" not in result

#####edge case
def test_preprocess_query_empty():
    result = query_process.preprocess_query("", lowercase=True)
    assert result == ""

def test_preprocess_query_only_stopwords():
    result = query_process.preprocess_query("the a an on", lowercase=True)
    # Assuming stopwords are removed
    assert result == ""

#####

# Test vector encoding (dynamic shape)
def test_query_vector_encoding():
    query = "best universities australia"
    vector = query_process.Query_processor.encode(query)

    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1     # one-dimensional
    assert vector.shape[0] > 0        # length > 0

def test_query_vector_empty_input():
    vector = query_process.Query_processor.encode("")
    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1 and vector.shape[0] > 0


