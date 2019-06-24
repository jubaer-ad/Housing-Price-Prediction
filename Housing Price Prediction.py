#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[12]:


import numpy as np


# In[46]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# In[87]:


df_train.columns


# In[42]:


df_test.head()


# 

# In[86]:





# In[ ]:





# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


plt.scatter(df_train['PoolArea'],df_train['SalePrice'])


# In[139]:


x = df_train.drop(['Id','SalePrice'], axis=1)
y = df_train['SalePrice']
x.columns


# In[140]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
for i in c:
    
    s = le.fit_transform(x[i].astype(str))
    x[i] = s
x.head()
#from sklearn.preprocessing import OneHotEncoder
#neHotEncoder().fit_transform(x)


# In[141]:


from sklearn.model_selection import train_test_split


# In[142]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[143]:


x_train.head()


# In[ ]:





# In[144]:


from sklearn.linear_model import LinearRegression


# In[145]:


clf = LinearRegression()


# In[147]:


clf.fit(x_train, y_train)


# In[148]:


clf.score(x_test,y_test)


# In[157]:


clf.predict(x_test)


# In[159]:


y_test


# In[ ]:




