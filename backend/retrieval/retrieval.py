#!/usr/bin/env python
# coding: utf-8

from query_process import preprocess_query, Query_processor
import faiss
import numpy as np
import pickle
import os



index_path = os.path.abspath("../../data/processed_texts/vector_index.bin")
sentences_path = os.path.abspath("../../data/processed_texts/sentences.pkl")
index = faiss.read_index(index_path)
sentences = pickle.load(open(sentences_path, "rb"))

def retrieve_top_k_documents(query, k=5):

    query_processed = preprocess_query(query) 
    query_vector = Query_processor.encode(query_processed)

    query_vector = np.array([query_vector], dtype=np.float32)
    
    distances, indices = index.search(query_vector, k)
    results = [(sentences[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    query = "how can i find accommodation?"
    top_k_results = retrieve_top_k_documents(query, k=5)

    print("\nTop K Retrieved Documents:")
    for i, (sentence, score) in enumerate(top_k_results):
        print(f"{i+1}. {sentence} (Score: {score:.4f})")









