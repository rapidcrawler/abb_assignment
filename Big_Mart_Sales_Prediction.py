#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[3]:


import pandas as pd
import numpy as np


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


# # Load Data

# In[99]:


# Load the datasets
train_df = pd.read_csv("./assets/ip/train_v9rqX0R.csv")
print(f"Train dataset size: {train_df.shape}")
display(train_df.info())
train_df.head()


# # EDA

# ## Summary Stats

# In[102]:


missing_values = train_df.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]
missing_values


# In[104]:


display(train_df['Item_Type'].value_counts()); print("\n\n")
display(train_df['Outlet_Identifier'].value_counts()); print("\n\n")
display(train_df['Outlet_Size'].value_counts(normalize=True)*100); print("\n\n")
display(train_df['Outlet_Location_Type'].value_counts(normalize=True)*100); print("\n\n")
display(train_df['Outlet_Type'].value_counts()); print("\n\n")


# ## Combine train and test for consistent encoding

# In[107]:


# Reload test and sample submission files after environment reset
test_df = pd.read_csv("./assets/ip/test_AbJTz2l.csv")
submission_df = pd.read_csv("./assets/ip/sample_submission_8RXa3c6.csv")


# In[109]:


# Combine train and test for consistent encoding
train_df['source'] = 'train'
test_df['source'] = 'test'
test_df['Item_Outlet_Sales'] = np.nan
combined = pd.concat([train_df, test_df], ignore_index=True)


# ## Data Cleaning

# In[112]:


# Handle missing values
combined['Item_Weight'].fillna(combined['Item_Weight'].mean(), inplace=True)
combined['Outlet_Size'].fillna('Medium', inplace=True)

# Normalize inconsistent values
combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'
})


# ### Categorical Encoding

# In[114]:


# Encode categorical features
categorical_cols = combined.select_dtypes(include='object').columns.drop(['Item_Identifier', 'source'])
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined[categorical_cols] = encoder.fit_transform(combined[categorical_cols])

# Split back
train_encoded = combined[combined['source'] == 'train'].drop(columns=['source'])
test_encoded = combined[combined['source'] == 'test'].drop(columns=['source', 'Item_Outlet_Sales'])


# In[116]:


# Prepare features and target
X = train_encoded.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'])
y = train_encoded['Item_Outlet_Sales']


# ## Train Test Split

# In[97]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# In[118]:


# Train-test split for validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# ## RF Regressor

# In[152]:


from sklearn.ensemble import RandomForestRegressor


# In[154]:


# Model: Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_valid)
rf_regressor_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
rf_regressor_rmse


# In[155]:


print(f"RF Regressor RMSE: {rf_regressor_rmse}")


# ## XGBoost Regressos

# In[164]:


# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
xgboost_rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Train RMSE: {xgboost_rmse}")


# In[162]:


# Predict on test set
test_features = test_encoded.drop(columns=['Item_Identifier'])
test_predictions = model.predict(test_features)


# # Submission Preparation

# In[124]:


# Prepare final submission DataFrame
submission_df = test_encoded[['Item_Identifier', 'Outlet_Identifier']].copy()
submission_df['Item_Outlet_Sales'] = test_predictions
submission_df.head()


# In[ ]:




