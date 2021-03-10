#!/usr/bin/env python
# coding: utf-8

# In[173]:


import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import pairwise, pairwise_distances
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


# In[119]:


dataset = pd.read_csv("data/dataset_split/train.csv", index_col=0)
dataset.head()


# In[120]:


model = SentenceTransformer("stsb-distilbert-base")


# In[121]:


reviews = list(map(str, dataset["review_text"].values))
review_summaries = list(map(str, dataset["review_summary"].values))


# In[123]:


dataset["review_vector"] = model.encode(reviews, show_progress_bar=True, device="cuda").tolist()


# In[125]:


dataset['review_summary_vector'] = model.encode(review_summaries, show_progress_bar=True, device="cuda").tolist()


# In[126]:


items_review_embeddings = dataset[['item_id', 'review_vector']].groupby('item_id').agg(lambda x: np.array(x.values.tolist()).mean(0).tolist())
items_review_summary_embeddings = dataset[['item_id', 'review_summary_vector']].groupby('item_id').agg(lambda x: np.array(x.values.tolist()).mean(0).tolist())

items_review_embeddings = pd.DataFrame(items_review_embeddings['review_vector'].to_list(), index=items_review_embeddings.index)
items_review_summary_embeddings = pd.DataFrame(items_review_summary_embeddings['review_summary_vector'].to_list(), index=items_review_summary_embeddings.index)


# In[127]:


items_categories = dataset[['item_id', 'category']].groupby('item_id').agg(pd.Series.mode)
items_categories_onehot = pd.get_dummies(items_categories)


# In[128]:


features = items_review_embeddings.join(items_review_summary_embeddings, how='left', lsuffix='_review', rsuffix='_review_summary')
features = features.join(items_categories_onehot, how='left', rsuffix='_category')
features.head()


# In[167]:


from sklearn.decomposition import PCA
pca = PCA(100)
features_pca = pd.DataFrame(pca.fit_transform(features), index = features.index)


os.makedirs('data/predictions', exist_ok=True)
for metric in ['euclidean', 'cosine', 'manhattan']:
    recomendations = {}
    for user_id, row in tqdm(dataset.groupby('user_id')):
        items_rented = set(row['item_id'])
        mean = features.loc[items_rented].mean(axis=0)
        mean = pca.transform([mean])[0]
        
        distances = pairwise_distances([mean], features_pca.values, metric=metric, n_jobs=-1)[0]
        maximum, minimum = (np.max(distances), np.min(distances))

        distances = (distances - minimum) / (maximum - minimum)
        recomendations[user_id] = dict(zip(map(str, features_pca.index), 1 - distances))
        
    with open(f'data/predictions/knn_{metric}.json', 'w') as f:
        json.dump(recomendations, f)

    


# In[ ]:




