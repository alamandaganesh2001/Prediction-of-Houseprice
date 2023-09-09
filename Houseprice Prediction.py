#!/usr/bin/env python
# coding: utf-8

# # ALAMANDA GANESH
# # TASK-1: House Price Prediction

# In[ ]:


#IMPORT NECESSARY LIBRARIES 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#FOR MACHINE LEARNING
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('HousePricePrediction.csv')
df
df.head(10)


# # DATA PREPROCESSING

# In[4]:


obj = (df.dtypes =='object')
object_cols = list(obj[obj].index)
print('categorical variables',len(object_cols))


# In[19]:


Int = (df.dtypes == 'int')
Int_cols = list(Int[Int].index)
print('Integers variables', len(Int_cols))


# In[6]:


Float = (df.dtypes == 'float')
Float_cols = list(Float[Float].index)
print('Float Variables',len(Float_cols))


# # EXPLORATORY DATA ANALYSIS

# In[7]:


plt.figure(figsize=(13,7))
sns.heatmap(df.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)


# In[8]:


unique_values = []
for col in object_cols:
  unique_values.append(df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title("NO. OF UNIQUE VALUES IN CATEGORICAL VARIABLES")
plt.xticks(rotation = 15)
sns.barplot(x=object_cols,y=unique_values) 


# In[9]:


unique_values = []
for col in Int_cols:
  unique_values.append(df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title("NO. OF UNIQUE VALUES IN INTEGER VARIABLES")
plt.xticks(rotation = 15)
sns.lineplot(x=Int_cols,y=unique_values)


# In[10]:


unique_values = []
for col in Float_cols:
  unique_values.append(df[col].unique().size)
plt.figure(figsize=(10,6))
plt.title("NO. OF UNIQUE VALUES IN FLOAT VARIABLES")
plt.xticks(rotation = 15)
sns.lineplot(x=Float_cols,y=unique_values)


# In[11]:


fig = plt.figure(figsize=(30, 86))
fig.suptitle('Categorical Features: Distribution')
index = 1

for col in object_cols:
    y = df[col].value_counts()
    ax = fig.add_subplot(11, 4, index)
    ax.set_xticklabels(y.index, rotation=90)
    sns.barplot(x=y.index, y=y, ax=ax)
    index += 1
    ax.set_title(col)
    
    # Explicitly remove overlapping axes
    for ax in fig.axes:
        plt.sca(ax)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

plt.show()


# # DATA CLEANING

# In[12]:


df['SalePrice'] = df['SalePrice'].fillna(
    df['SalePrice'].mean()
)


# In[13]:


new_df = df.dropna()
new_df.isnull().sum()


# # ONE HOT ENCODER - FOR LABEL CATEGORICAL FEATURES AND SPLITTING DATASET INTO TRAINING AND TESTING

# In[14]:


s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print('catogorical Variables: ')
print(object_cols)
print('no.of catogorical Variables:',len(object_cols))


# In[15]:


one_hot_encoder = OneHotEncoder(sparse_output = False)
one_hot_cols = pd.DataFrame(one_hot_encoder.fit_transform(new_df[object_cols]))
one_hot_cols.index = new_df.index
one_hot_columns = one_hot_encoder.get_feature_names_out()
df_final = new_df.drop(object_cols, axis =1)
df_final = pd.concat([df_final,one_hot_cols],axis =1)


# In[16]:


'''Now splitting data into Training and Testing'''
X = df_final.drop(['SalePrice'], axis = 1)
Y = df_final['SalePrice']
X.columns = X.columns.astype(str)


# In[17]:


#splitting the training set into
#training and validation set
X_train, X_valid,Y_train,Y_valid = train_test_split(
    X,Y,train_size = 0.8,test_size = 0.2,random_state = 0
)

MODEL AND ACCURACY
LINEAR REGRESSION
# In[18]:


# Train the model
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)

# Make predictions
Y_pred = model_LR.predict(X_valid)

# Evaluate the model
mspe = mean_absolute_percentage_error(Y_valid, Y_pred)
print('Mean absolute percentage error:', mspe)

