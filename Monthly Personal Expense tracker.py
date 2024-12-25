#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[70]:


df = pd.read_csv('myExpenses1.csv')
df.head()


# In[71]:


df.info()


# In[72]:


df.describe()


# In[73]:


df.isnull().sum()


# In[74]:


# Drop rows where 'Time' is null
df = df.dropna(subset=['Time'])


# In[75]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')


# In[76]:


df['Hour'] = df['Date'].dt.hour


# In[77]:


def time_of_day(hour):
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

# Apply the function to create the 'Time_of_Day' column
df['Time_of_Day'] = df['Hour'].apply(time_of_day)


# In[78]:


print(df.columns)


# In[79]:


df = df.drop(columns=['Hour'])


# In[80]:


df.head()


# In[81]:


print(df.columns)


# In[82]:


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
print(df['Time'].unique())

df = df[pd.to_datetime(df['Time'], errors='coerce').notna()]

df['Hour'] = pd.to_datetime(df['Time']).dt.hour

le = LabelEncoder()
df['Time_of_Day'] = le.fit_transform(df['Time_of_Day'])



# In[83]:


df = pd.get_dummies(df, columns=['Category'], drop_first=True)

features = ['Amount', 'Hour'] + [col for col in df.columns if 'Category_' in col]


# In[84]:


X = df[features]  # Features
y = df['Time_of_Day']  # Target variable
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[85]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)



# In[86]:


y_pred = model.predict(X_test)


# In[87]:


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[88]:


df.head()


# In[ ]:





# In[ ]:




