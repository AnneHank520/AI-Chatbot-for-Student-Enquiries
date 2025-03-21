#!/usr/bin/env python
# coding: utf-8

import fitz
import spacy
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

def extract_text_from_pdf(file_path):
    text = ''
    reader = fitz.open(file_path)
    for page in reader:
        text += page.get_text('text') + '\n'
    return text.strip()

def preprocess_text(text, lowercase=True):
    if lowercase:
        text = text.lower()
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    # text = re.sub(r'http\S+|www\.\S+', '', text)  ## cancelling removing links
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]   
    return sentences


def text_to_vector_sentence_transformer(text):

    return sentence_transformer.encode(text)

# sentence_transformer.save('sentence_transformer.model')
# sentence_transformer = SentenceTransformer('sentence_transformer.model')

# print(sentences_vector[0:10])

# print(sentences_vector[0].shape)


if __name__ == "__main__":
    file_path = os.path.abspath("../../data/Resource.pdf")  # Change the file path here
    pdf_text = extract_text_from_pdf(file_path)

    # Initialize spaCy and Sentence Transformer
    nlp = spacy.load('en_core_web_sm')
    model_path = os.path.abspath("../../sentence_transformer.model")
    sentence_transformer = SentenceTransformer(model_path)

    sentences = preprocess_text(pdf_text)
    
    # Convert sentences to vectors
    sentences_vector = [text_to_vector_sentence_transformer(sentence) for sentence in sentences]

    # Initialize FAISS index and add vectors
    vector_dimension = sentences_vector[0].shape[0]
    vector_db = faiss.IndexFlatL2(vector_dimension)
    all_vectors = np.array(sentences_vector, dtype=np.float32)
    vector_db.add(all_vectors)
    index_path = os.path.abspath("../../data/processed_texts/vector_index.bin")
    sentences_path = os.path.abspath("../../data/processed_texts/sentences.pkl")

    # Save the FAISS index
    faiss.write_index(vector_db, index_path)
    print(f'Vector database saved to {index_path}')
    with open(sentences_path, "wb") as f:
        pickle.dump(sentences, f)
    print(f"✅ Sentences saved at {sentences_path}")