#!/usr/bin/env python
# coding: utf-8

# In[4]:


from glob import glob
import json
import pandas as pd
import numpy as np
import os


# In[5]:


files = glob("data/predictions/*")


# In[6]:


N = 60
os.makedirs("data/top_n", exist_ok=True)
for file_path in files:
    filename = file_path.split("/")[-1]

    df = pd.read_json(file_path, orient='index')
    col_indices = np.argsort(df.values, axis=-1)[:, -N:]
    recomendations = {}
    for user_id, col_indexes, values in zip(df.index, col_indices, df.values):
        recomendations[user_id] = dict(zip(df.columns[col_indexes], values[col_indexes]))

    with open(f"data/top_n/{filename}", 'w') as f:
        json.dump(recomendations, f)


# In[ ]:




