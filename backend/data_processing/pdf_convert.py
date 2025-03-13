#!/usr/bin/env python
# coding: utf-8

import fitz
import spacy
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def extract_text_from_pdf(file_path):
    text = ''
    reader = fitz.open(file_path)
    for page in reader:
        text += page.get_text('text') + '\n'
    return text.strip()
file_path = 'Resource.pdf'
pdf_text = extract_text_from_pdf(file_path)

# print(pdf_text[0:500])

nlp = spacy.load('en_core_web_sm')
def preprocess_text(text, lowercase=True):
    if lowercase:
        text = text.lower()
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    text = re.sub(r'http\S+|www\.\S+', '', text)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]   
    return sentences

sentences = preprocess_text(pdf_text)

sentence_transformer = SentenceTransformer('sentence-t5-large')
def text_to_vector_sentence_transformer(text):
    return sentence_transformer.encode(text)

# sentence_transformer.save('sentence_transformer.model')
# sentence_transformer = SentenceTransformer('sentence_transformer.model')


sentences_vector = [text_to_vector_sentence_transformer(sentence) for sentence in sentences]
# print(sentences_vector[0:10])

# print(sentences_vector[0].shape)

vector_dimension = sentences_vector[0].shape[0]
vector_db = faiss.IndexFlatL2(vector_dimension)
all_vectors = np.array(sentences_vector, dtype=np.float32)
vector_db.add(all_vectors)

faiss.write_index(vector_db, 'vector_index.bin')
print('Vector database saved to vector_index.bin')


