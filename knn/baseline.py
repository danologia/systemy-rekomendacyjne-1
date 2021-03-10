#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import json


# In[2]:


dataset = pd.read_csv("data/dataset_split/train.csv", index_col=0)


# In[18]:


indexes = dataset[dataset['class_rating']==1]['item_id'].value_counts()[:60].index
probs = 1 - np.arange(len(indexes)) / len(indexes)

mapping = dict(zip(indexes, probs))


# In[24]:


users = {
    user: mapping
    for user in set(dataset['user_id'])
}


# In[25]:


with open("data/baseline.json", 'w') as f:
    json.dump(users, f)


# In[ ]:




