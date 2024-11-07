#!/usr/bin/env python
# coding: utf-8

# # Emotion Detection Classification menggunakan Dataset Twitter Emotion Classification Task
# 
# Andara Najla Jilan - EWAKO

# In[1]:


import numpy as np 
import pandas as pd 


# In[2]:


df = pd.read_csv("Twitter_Emotion_Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


df["label"].value_counts()


# In[8]:


df["tweet"][1:10]


# ## **Preprocessing**

# In[9]:


import re
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>@'),"",raw_text)
    return cleaned_text


# In[11]:


df["tweet"] = df["tweet"].apply(remove_tags)


# In[12]:


df.head(5)


# In[14]:


#lower casing
df["tweet"] = df["tweet"].apply(lambda x: x.lower())


# In[16]:


#apply stopwords
from nltk.corpus import stopwords


sw_list = stopwords.words("indonesian")

df['tweet'] = df['tweet'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))


# In[17]:


df


# In[18]:


X = df.iloc[:,1:2]
y = df["label"]


# In[19]:


X.head()


# In[20]:


y.head()


# In[21]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)


# In[22]:


y


# In[23]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[24]:


X_train.shape


# In[25]:


y_test.shape


# ## **Text Vectorization**

# In[26]:


#Applying bag of words
from sklearn.feature_extraction.text import CountVectorizer


# In[27]:


cv = CountVectorizer()


# In[28]:


X_train_bow = cv.fit_transform(X_train['tweet']).toarray()
X_test_bow = cv.transform(X_test['tweet']).toarray()


# In[29]:


X_train_bow.shape


# ## **Modelling**

# In[30]:


from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

mnb.fit(X_train_bow,y_train)


# In[31]:


y_pred = mnb.predict(X_test_bow)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)


# In[32]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(multi_class='multinomial')

lr.fit(X_train_bow,y_train)


# In[33]:


y_pred = lr.predict(X_test_bow)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)


# # Kesimpulan
# 
# 1. Akurasi yang didapatkan pada naive bayes sebesar 0.6174801362088536
# 2. Akurasi dari Logistic Regression sebesar 0.64472190692395
# 3. Berdasarkan kedua model, akurasi yang didapatkan sekitar 60% benar
# 

# In[ ]:




