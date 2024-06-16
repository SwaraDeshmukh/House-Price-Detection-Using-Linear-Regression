#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import zipfile
import tempfile
import os


# In[48]:


def load_dataset(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_ref.extractall(tmpdirname)
            dataset = pd.read_csv(os.path.join(tmpdirname, file_name))
    return dataset


# In[49]:


zip_path = 'D:\Internship\House Price Dataset.zip' 
file_name = 'Housing.csv'  
house_data = load_dataset(zip_path, file_name)


# In[50]:


print("Original column names in the dataset:")
print(house_data.columns)


# In[51]:


house_data.columns = house_data.columns.str.lower().str.strip()
print("Standardized column names in the dataset:")
print(house_data.columns)


# In[52]:


numeric_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
house_data = house_data[numeric_columns]


# In[53]:


if house_data.isnull().sum().any():
    house_data = house_data.dropna()


# In[54]:


house_data = house_data.apply(pd.to_numeric, errors='coerce')


# In[55]:


target_variable = 'price'
features = house_data.drop(columns=[target_variable])
target = house_data[target_variable]


# In[56]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[57]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[58]:


y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


# In[59]:


train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)


# In[60]:


print(f'Training MSE: {train_mse}')
print(f'Testing MSE: {test_mse}')
print(f'Training R^2: {train_r2}')
print(f'Testing R^2: {test_r2}')


# In[61]:


def predict_price(features):
    features_df = pd.DataFrame([features])
    prediction = model.predict(features_df)
    return prediction[0]


# In[62]:


example_features = X_test.iloc[0].to_dict()
print("Example features for prediction:")
print(example_features)


# In[63]:


predicted_price = predict_price(example_features)
print(f'Predicted price: {predicted_price}')
print(f'Actual price: {y_test.iloc[0]}')


# In[ ]:




