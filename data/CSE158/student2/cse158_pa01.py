#!/usr/bin/env python
# coding: utf-8

# In[58]:


# Name: Yifei Ning
# PID: A14508232


# In[59]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[60]:


# with open('amazon_reviews_us_Gift_Card_v1_00.tsv') as tsv_file:
#     text = csv.reader(tsv_file, delimiter= ' ')
#     for row in text:
#         print(len(row))
fp = 'amazon_reviews_us_Gift_Card_v1_00.tsv'
text = pd.read_csv(fp, header = 0, sep='\t')
text.head(3).T


# In[3]:


text.loc[0, :]


# # Q_01

# In[4]:


# dist of ratings
dist_stars = text.star_rating
ax_1 = dist_stars.value_counts().plot(
    'bar', title = 'distribution of stars')
ax_1.set_xlabel("star rated")
ax_1.set_ylabel("numbers")
dist_stars.value_counts()


# # Q_02

# In[5]:


dist_stars_ver = text.loc[text.verified_purchase == 'Y', :].star_rating
dist_stars_nonver = text.loc[text.verified_purchase == 'N', :].star_rating
ax_2 = dist_stars_ver.value_counts().plot(
    'bar', 
    title = 'distribution of stars of the verified')
ax_2.set_xlabel("star rated")
ax_2.set_ylabel("numbers")
plt.show()

ax_3 = dist_stars_nonver.value_counts().plot(
    'bar', title = 'distribution of stars of the non-verified')
ax_3.set_xlabel("star rated")
ax_3.set_ylabel("numbers")
plt.show()


# # Q_03

# ### θ_0: for a non-verified review of 0 characters, the rating is likely to be θ_1 or approx. 4.85. 
# 
# ### θ_1: for verified review, the rating is likely to be 4.99e-02 higher than those non-verified of the same length.
# 
# ### θ_2: for each additional character in the review, the rating is likely to be -1.25e-03 lower.
# 
# ### Since the coefficient of "review length" is negative, that means nagative reviews tend to have longer length.

# In[6]:


preclean_data = text.loc[:, ['verified_purchase', 'review_body', 'star_rating']]
preclean_data.loc[:, 'verified_purchase'] = (preclean_data.verified_purchase == 'Y').astype(np.int64)
preclean_data.loc[:, 'review_body'] = preclean_data.loc[:, 'review_body'].astype(np.str)
preclean_data.loc[:, 'review_length'] = preclean_data.review_body.apply(len)
clean_data = preclean_data[['verified_purchase', 'review_length', 'star_rating']]
clean_data.head(10)


# In[7]:


def design_matrix(feat):
    out = np.zeros((feat.shape[0], feat.shape[1] + 1))
#     print(out)
    for i in range(len(feat)):
        out[i] = np.array([1] + list(feat[i]))
    return out


# In[8]:


# get the design matrix and predictor
X = clean_data[['verified_purchase', 'review_length']].values
X = design_matrix(X)
y = clean_data.star_rating.values


theta, res, ranks, s = np.linalg.lstsq(X, y)
theta


# # Q_04

# ### θ_0: for a non-verified review of 0 characters, the rating is likely to be 4.578143. 
# 
# ### θ_1: how much higher (0.16793392) the verified rating is likely to be than those non-verified.
# 
# ### The review_length as a feature might be highly correlated with the prediction of star ratings and verified_purchase is not that correlated with the star ratings. If review_length is removed and all the weights of the coefficient previously on that feature will go to verified_purchased, resulting in a unreliable results. Plus, "verified_purchase" is categorical and should be processed by one hot encoding. Simply removing this feature will make the prediction worse.

# In[10]:


X_1 = clean_data[['verified_purchase']].values
X_1 = design_matrix(X_1)
y_1 = clean_data.star_rating.values

theta, res, ranks, s = np.linalg.lstsq(X_1, y_1)
theta

# def model_training(X, y, model):
#     return model(X, y)

# model_training(X_1, y_1, np.linalg.lstsq)


# # Q_05

# ### The model's MSE on the training set is approx. 0.655 ; on the test set is approx. 0.972.

# In[11]:


def mse(input_labels, output_labels):
    assert len(input_labels) == len(output_labels), 'mis-match of length'
    
    out = np.mean((input_labels - output_labels) ** 2)
    return out


# In[12]:


TEST_SIZE = 0.1
X_train, X_test, y_train, y_test = train_test_split(clean_data[['verified_purchase']].values,
                                                    clean_data.star_rating.values,
                                                    test_size = TEST_SIZE, 
                                                    shuffle = False)
X_train = design_matrix(X_train)
X_test = design_matrix(X_test)


# In[13]:


theta_train, res_train, ranks_train, s_train = np.linalg.lstsq(X_train, y_train)
theta_train


# In[14]:


train_output_labels = X_train @ theta_train
mse(y_train, train_output_labels)
# train_output_labels


# In[15]:


test_output_labels = X_test @ theta_train
mse(y_test, test_output_labels)


# # Q_06

# ### MAE: 0.6221007247297486
# ### R^2: -0.04811587359357783

# In[16]:


def mae(input_labels, output_labels):
    out = np.mean(abs(input_labels - output_labels))
    return out

def r_sq(input_labels, output_labels):
    MSE = mse(input_labels, output_labels)
    VAR = np.var(input_labels)
    out = 1 - MSE / VAR
    return out
    


# In[17]:


# run on test set
theta_test, res_test, ranks_test, s_test = np.linalg.lstsq(X_test, y_test)

test_output_labels = X_test @ theta_train


# In[18]:


mae(y_test, test_output_labels)


# In[37]:


r_sq(y_test, test_output_labels)


# # Q_08
# 
# ### accuracy_score on testing set:  0.5597734475085968
# ### proportion of positive predictions on testing: 0.9989886049490931
# ###  proportion of positive predictions on training: 0.9996478846859805
# ### proportion of positive labels:
# - proportion of positive labels in test set:  0.5595711684984155
# - proportion of positive labels in training set:  0.9513856112197424
# -  proportion of positive labels in training + test set:  0.9122041669476098
# 
# 
# 

# In[53]:


features = clean_data.loc[:, ['star_rating', 'review_length']].values
labels = clean_data.verified_purchase.values

X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size = 0.1, shuffle = False)

X_tr = design_matrix(X_tr)
X_te = design_matrix(X_te)


# In[54]:


# fitting the model
clf = LogisticRegression().fit(X_tr, y_tr)


# In[56]:


np.mean(clf.predict(X_tr))


# In[41]:


y_pred = clf.predict(X_te)
print('accuracy_score: ', accuracy_score(y_pred, y_te))


# In[42]:


def report_labels_percentage(y, type_):
    positive_labels = np.sum(y)
    out = positive_labels / len(y)
    print('proportion of positive labels in ' + type_ + ' set: ', out)
#     return out;


# In[43]:


report_labels_percentage(y_te, 'test')


# In[44]:


report_labels_percentage(y_tr, 'training')


# In[45]:


report_labels_percentage(clean_data.verified_purchase.values, 'training + test')


# In[46]:


report_labels_percentage(y_pred, 'prediction')


# # Q_09

# ### I add one more feature "product_id" to the linear predictor in Q_08. Since, people are likely to endorse good books and give good comments/ratings. If a book is good, lots of people may find that in common and more likely the rating is verified. 
# 
# ### Then my predictor becomes: 
# - p(review is verified) ≃ σ(θ0 + θ1 × [star rating] + θ2 × [review length] +  θ3 × [id == 1] + ... + θn × [id == n])
# 
# ### I will use label encoding and one hot encoding to tranform the categorical columns.
# 
# ### training accuracy:  0.96(approx.)
# ### test accuracy:  0.89 (approx.)

# In[47]:


label_encode = LabelEncoder()

one_hot = text.product_id.value_counts()
selected_df = clean_data
selected_df.loc[:, 'product_id'] = text.loc[:, ['product_id']].apply(label_encode.fit_transform)
selected_df.loc[:, 'verified_purchase_new'] = selected_df.loc[:, 'verified_purchase']
selected_df = selected_df.drop(['verified_purchase'], 1)

# check what's the column index for the cate feat col
one_hot_encode = OneHotEncoder(categorical_features = [-2])

encoded = one_hot_encode.fit_transform(selected_df)
final_matrix = encoded.toarray()
final_matrix = design_matrix(final_matrix)
final_matrix


# In[48]:



feats = final_matrix[:, :-1]
labs = final_matrix[:, -1]

X_tr, X_te, y_tr, y_te = train_test_split(feats, labs, test_size = 0.1, shuffle = False)


# In[49]:


clf_09 = LogisticRegression().fit(X_tr, y_tr)


# In[50]:


print('training accuracy: ', np.mean(clf_09.predict(X_tr) == y_tr))


# In[51]:


print('test accuracy: ', np.mean(clf_09.predict(X_te) == y_te))


# In[ ]:




