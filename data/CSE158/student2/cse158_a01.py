#!/usr/bin/env python
# coding: utf-8

# In[5]:


import re
import gzip
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import balanced_accuracy_score

def readGz(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.loads(l)


# In[6]:


data = []
for l in readGz("renttherunway_final_data.json.gz"):
    data.append(l)
    
df = pd.DataFrame(data)


# In[7]:


print(df.shape)
df.head()


# In[8]:


def modify_height(row):
    height = row['height']
    if isinstance(height, float):
        return height
    lst = re.findall(r'\d+', height)
    return (int(lst[0]) * 12 + int(lst[1])) * 2.54

def modify_weight(row):
    weight = row['weight']
    if isinstance(weight, float):
        return weight
    return int(re.findall(r'\d+', weight)[0])

modified_height = df.apply(modify_height, axis = 1)
modified_weight = df.apply(modify_weight, axis = 1)

df['height(cm)'] = modified_height
df['modified_weight'] = modified_weight

df = df.drop(['height', 'weight'], axis = 1)


# In[9]:


for val in df.columns:
    print('There are: ', sum(df[val].isnull()), ' null values in ', val)


# In[10]:


df.dropna(subset=['age','body type','bust size','rating','rented for','height(cm)','modified_weight'],inplace = True)


# In[11]:


for val in df.columns:
    print('There are: ', sum(df[val].isnull()), ' null values in ', val)


# In[12]:


print(df.shape)
df.head()


# In[13]:


df['age'] = df['age'].astype(float)
df['height(cm)'] = df['height(cm)'].astype(float)
df['size'] = df['size'].astype(float)
df['modified_weight'] = df['modified_weight'].astype(float)
df['rating'] = df['rating'].astype(float)


# In[14]:


df['fit'].value_counts() / np.sum(df['fit'].value_counts())


# In[15]:


df['fit'] = df['fit'].map({'fit': 1, 'small': 2,'large':3})


# In[16]:


print(df.groupby('fit')['rating'].apply(np.mean))
ax = df.groupby('fit')['rating'].apply(np.mean).plot(kind = 'bar');
plt.title('ratings vs. fit')
plt.ylabel('ratings');


# In[17]:


df_fit = df[df['fit']==1]
df_small = df[df['fit']==2]
df_large = df[df['fit']==3]


# In[18]:


df['rented for'].value_counts()


# In[19]:


df['rented for'] = df['rented for'].map({'wedding': 1, 'formal affair': 2,'party':3,
                                         'everyday':4,'work':5,'other':6,'date':7,'vacation':8,
                                         'party: cocktail':9})


# In[20]:


df['body type'].value_counts()


# In[21]:


df['body type'] = df['body type'].map({'hourglass': 1, 'athletic': 2,'petite':3,
                                         'pear':4,'straight & narrow':5,'full bust':6,'apple':7})


# In[22]:


bust_size_count = df['bust size'].value_counts()
encoder={}
for i in range (len(bust_size_count)):
    encoder[bust_size_count.index[i]] = i
df['bust size'] = df['bust size'].map(encoder)


# In[23]:


id_count = df['user_id'].value_counts()
encoder={}
for i in range (len(id_count)):
    encoder[id_count.index[i]] = i
df['user_id'] = df['user_id'].map(encoder)


# In[24]:


df.head()


# ## EDA

# #### general statistics

# In[25]:


df.drop(['body type','fit', 'rented for', 'user_id'], axis = 'columns').describe()


# ### Data type

# In[26]:


data_column_type = pd.Series({'age':'numerical','body type':'categorical',
                             'bust size':'nominal','category':'categorical','fit':'categorical',
                             'height':'numerical','item_id':'nominal','rating':'ordinal',
                             'rented for':'categorical','review_date':'date','review_summary':'text',
                             'review_text':'text','size':'numerical','user_id':'nominal',
                             'weight':'numerical'}, name = 'type of data')


# In[27]:


#### general statistics

df.describe()

### Data type

data_column_type = pd.Series({'age':'numerical','body type':'categorical',
                             'bust size':'nominal','category':'categorical','fit':'categorical',
                             'height':'numerical','item_id':'nominal','rating':'ordinal',
                             'rented for':'categorical','review_date':'date','review_summary':'text',
                             'review_text':'text','size':'numerical','user_id':'nominal',
                             'weight':'numerical'}, name = 'type of data')


# In[28]:


data_column_type.to_frame()


# In[29]:


data_column_type.value_counts().plot(kind = 'bar')


# #### rating only contains 2,4,6,8,10

# In[30]:


(df.rating.value_counts() / np.sum(df.rating.value_counts())).plot(kind = 'bar')
ax = plt.title('rating distributions');
plt.xlabel('ratings');
plt.ylabel('percentage');
df.rating.value_counts() / np.sum(df.rating.value_counts())


# In[31]:


df.rating.plot(kind='hist');
ax = plt.title('rating distributions');
plt.xlabel('ratings');


# ##### Customers who found the clothing 'fit' seldom gives lower score

# In[32]:


df.head()


# In[33]:


fg,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,10))
ax1.set_ylim(0,80000)
ax2.set_ylim(0,80000)
ax3.set_ylim(0,80000)
df.loc[df.fit==1,['rating']].groupby('rating').plot(kind='hist',ax=ax1, color='g')
df.loc[df.fit==2,['rating']].groupby('rating').plot(kind='hist',ax=ax2, color='r')
df.loc[df.fit==3,['rating']].groupby('rating').plot(kind='hist',ax=ax3, color = 'b')
ax1.set_xlabel('fit = fit (1)');
ax2.set_xlabel('fit = small (2)');
ax3.set_xlabel('fit = large (3)');

# df.loc[df.fit==1,['rating']].value_counts()


# #### rent cloth is popular among younger people

# In[66]:


x_mean = np.mean(df['age'])
x_std = np.std(df['age'])
x_median = np.median(df['age'])


# In[67]:


x = df['age']#.plot(kind='hist', bin_size = 5)
plt.title('density distribution of ages');
plt.xlabel('age');

n, bins, patches = plt.hist(x, 60, density=True, facecolor='g', alpha=0.75)
plt.axvline(x_mean, color='k', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(x_mean*1.1, max_ylim*0.9, 'Mean(shown): {:.2f} Median: {:.2f} SD: {:.2f}'.format(x_mean,x_median,x_std));


# In[68]:


# df.groupby(['age'])[['age', 'rating']].apply(lambda x: 2 * (x // 5))


# ### Heatmap-fit

# In[69]:


corr_map = df.corr()
plt.subplots(figsize=(16,16))
sns.heatmap(corr_map, vmax=0.9, square=True, fmt = '.3f', annot = True, linewidths = .5)


# ### train-test split

# In[70]:


from sklearn.model_selection import train_test_split

df_train,df_test = train_test_split(df, test_size=0.33)


# In[71]:


from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(handle_unknown='ignore')
onehot.fit(df_train[['body type','rented for']])
def after_onehot(df):
    copy = df.copy(deep=True)
    onehot_transformed = onehot.transform(copy[['body type','rented for']]).toarray()
    copy = copy.drop(['body type','rented for'],axis=1)
    return np.concatenate((copy.values,onehot_transformed),axis=1)


# In[72]:


df_base_train = df_train[['fit','age','body type','rating','rented for','size','height(cm)','modified_weight']]
df_base_test = df_test[['fit','age','body type','rating','rented for','size','height(cm)','modified_weight']]

df_base_train = after_onehot(df_base_train)
df_base_test = after_onehot(df_base_test)


# ### baseline

# In[73]:


mod = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod.fit(np.delete(df_base_train,0,axis=1),df_base_train[:,0])


# In[74]:


pred = mod.predict(np.delete(df_base_test,0,axis=1))


# In[75]:


#ACC
sum(pred==df_base_test[:,0])/len(df_test)


# In[76]:


#BER
balanced_accuracy_score(pred, df_base_test[:,0])


# ### TF-IDF

# In[77]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_features=10000)
X_train = vectorizer.fit_transform(df_train['review_text'])


# In[78]:


X_train.shape


# In[79]:


mod_tfidf = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod_tfidf.fit(X_train,df_train['fit'])


# In[80]:


X_test = vectorizer.transform(df_test['review_text'])
pred_tfidf = mod_tfidf.predict(X_test)


# In[81]:


#For later emsemble learning pp
pred_tfidf_train= mod_tfidf.predict(X_train)


# In[82]:


#ACC
sum(pred_tfidf==df_test['fit'])/len(df_test)


# In[83]:


#BER
balanced_accuracy_score(pred_tfidf, df_test['fit'])


# ### TF-IDF + Baseline

# In[84]:


X_concat = np.concatenate([np.delete(df_base_train,0,axis=1),X_train.toarray()],axis = 1)


# In[85]:


X_concat_test = np.concatenate([np.delete(df_base_test,0,axis=1),X_test.toarray()],axis = 1)


# In[86]:


mod_concat = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod_concat.fit(X_concat,df_train['fit'])


# In[87]:


pred_concat = mod_concat.predict(X_concat_test)


# In[88]:


#ACC
sum(pred_concat==df_test['fit'])/len(df_test)


# In[89]:


#BER
balanced_accuracy_score(pred_concat, df_test['fit'])


# ### TF-IDF + Baseline Prediction

# In[90]:


pred_train = mod.predict(np.delete(df_base_train,0,axis=1))


# In[91]:


X_concat_pred= np.concatenate((X_train.toarray(), pred_train.reshape(-1,1)), axis=1)


# In[92]:


x_concat_pred_test = np.concatenate((X_test.toarray(), pred.reshape(-1,1)), axis=1)


# In[93]:


mod_concat_pred = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod_concat_pred.fit(X_concat_pred,df_train['fit'])


# In[94]:


pred_p_concat = mod_concat_pred.predict(x_concat_pred_test)


# In[95]:


#ACC
sum(pred_p_concat==df_test['fit'])/len(df_test)


# In[96]:


#BER
balanced_accuracy_score(pred_p_concat, df_test['fit'])


# ### TF-IDF Prediction + Baseline Prediction

# In[97]:


X_pp = np.vstack((pred_tfidf_train,pred_train)).T
X_pp


# In[42]:


X_pp_test = np.vstack((pred_tfidf,pred)).T


# In[43]:


mod_pp = linear_model.LogisticRegression(C=1.0,class_weight='balanced')
mod_pp.fit(X_pp,df_train['fit'])


# In[44]:


pred_pp = mod_pp.predict(X_pp_test)


# In[45]:


#ACC
sum(pred_pp==df_test['fit'])/len(df_test)


# In[46]:


#BER
balanced_accuracy_score(pred_pp, df_test['fit'])


# In[ ]:




