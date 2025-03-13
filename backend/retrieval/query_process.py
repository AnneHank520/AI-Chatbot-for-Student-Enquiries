#!/usr/bin/env python
# coding: utf-8

# In[4]:


import re
import spacy
from sentence_transformers import SentenceTransformer
import os



# In[ ]:


nlp = spacy.load('en_core_web_sm')
def preprocess_query(query, lowercase=True):
    if lowercase:
        query = query.lower()
    query = re.sub(r"[^a-z0-9\s]", "", query)
    query = re.sub(r'\s+', ' ', query).strip()
    doc = nlp(query)
    query_tokens = ' '.join([token.text for token in doc if not token.is_stop])
    return query_tokens 


# In[2]:


query = "  What are the best universities in Australia? "
pre_processed_query = preprocess_query(query)
print(pre_processed_query)



# In[5]:


model_path = os.path.abspath("../../sentence_transformer.model")
Query_processor = SentenceTransformer(model_path)


# In[6]:


query_vector = Query_processor.encode(pre_processed_query)
print(query_vector)
print(query_vector.shape)




# In[ ]:




