#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mwclient


# In[2]:


import mwclient
import time

site = mwclient.Site('en.wikipedia.org')
page = site.pages['Bitcoin']


# In[3]:


revs = list(page.revisions())


# In[4]:


revs[0]


# In[5]:


revs = sorted(revs, key=lambda rev: rev["timestamp"]) 


# In[6]:


revs[0]


# In[7]:


pip install --upgrade transformers


# In[8]:


from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1
    return score


# In[9]:


edits = {}

for rev in revs:        
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments=list(), edit_count=0)
    
    edits[date]["edit_count"] += 1
    
    comment = rev.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))


# In[10]:


from statistics import mean

for key in edits:
    if len(edits[key]["sentiments"]) > 0:
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0
    
    del edits[key]["sentiments"]


# In[11]:


import pandas as pd

edits_df = pd.DataFrame.from_dict(edits, orient="index")


# In[12]:


edits_df


# In[13]:


edits_df.index = pd.to_datetime(edits_df.index)


# In[14]:


from datetime import datetime

dates = pd.date_range(start="2009-03-08",end=datetime.today())


# In[15]:


dates


# In[16]:


edits_df = edits_df.reindex(dates, fill_value=0)


# In[17]:


edits_df


# In[18]:


rolling_edits = edits_df.rolling(30, min_periods=30).mean()


# In[20]:


rolling_edits


# In[22]:


rolling_edits = rolling_edits.dropna()


# In[23]:


rolling_edits


# In[24]:


rolling_edits.to_csv("wikipedia_edits.csv")


# In[ ]:




