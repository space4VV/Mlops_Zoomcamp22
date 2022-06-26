#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[1]:


import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[3]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[4]:


df = read_data('./data/fhv_tripdata_2021-02.parquet')


# In[5]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[20]:


y_pred


# In[11]:


print('mean of the predictions = ',sum(y_pred)/len(y_pred))


# In[17]:


year=2021
month=2
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[18]:


df.columns


# In[19]:


df['ride_id'] 


# In[21]:


df_final_pq = pd.DataFrame()
df_final_pq['ride_id'] = df['ride_id']
df_final_pq['predictions'] = y_pred


# In[22]:


df_final_pq.head()


# In[23]:


#writing the results as a parquet file:
df_final_pq.to_parquet(
    './df_results.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[14]:


df.index


# In[24]:


get_ipython().system('jupyter nbconvert --to script starter.ipynb')


# In[ ]:




