#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os


import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


df = pd.read_csv('myExpenses1.csv')
df.head()


# In[46]:


df.info()


# In[47]:


df.describe()


# In[48]:


df.isnull().sum()


# In[49]:


df['Category'].fillna('alone', inplace=True)


# In[50]:


# Drop rows where 'Time' is null
df = df.dropna(subset=['Time'])


# In[51]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')


# In[52]:


df['Hour'] = df['Date'].dt.hour
df['Amount'] = df['Amount']*10


# In[53]:


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


# In[54]:


print(df.columns)


# In[55]:


df = df.drop(columns=['Hour'])


# In[56]:


df['Amount_Category'] = pd.cut(df['Amount'], bins=[0, 50, 100, float('inf')], labels=["Low", "Medium", "High"])


# In[57]:


df.head()


# In[58]:


print(df.columns)


# In[59]:


df = pd.get_dummies(df, columns=['Category'])


# In[60]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Time_of_Day'] = le.fit_transform(df['Time_of_Day'])


# In[61]:


# Function to classify expenses
def classify_expense(row):
    if row['day'] in ['Monday', 'Wednesday', 'Friday'] or row['Time_of_Day'] == 'Night':
        return 'Less Essential'
    else:
        return 'Essential'

# Apply the classification logic
df['Class'] = df.apply(classify_expense, axis=1)



# In[62]:


df.head()


# In[63]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# Define features (X) and target (y)
X = df[['day', 'Time_of_Day', 'Amount','Category_alone','Category_friend']]  # Features
y = df['Class']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical columns using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['day', 'Time_of_Day','Category_friend','Category_alone']),
        ('num', 'passthrough', ['Amount'])
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Train a Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[41]:


from sklearn.model_selection import train_test_split

X = df.drop(['Amount_Category', 'Amount'], axis=1)  # Features (exclude the target and Amount columns)
y = df['Amount_Category']  # Target variable

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),  # Scale numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # One-hot encode categorical features
    ]
)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize and train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


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
