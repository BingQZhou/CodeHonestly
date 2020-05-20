#!/usr/bin/env python
# coding: utf-8

# In[971]:


# Name: Yifei Ning
# PID: A14508232


# In[972]:


from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[973]:


# df1 = pd.DataFrame(arff.loadarff('./data/1year.arff')[0])
# df2 = pd.DataFrame(arff.loadarff('./data/2year.arff')[0])
# df3 = pd.DataFrame(arff.loadarff('./data/3year.arff')[0])
# df4 = pd.DataFrame(arff.loadarff('./data/4year.arff')[0])

# In[974]:


def clean_data(df):
    df = df.dropna()
    df.iloc[:, -1] = df.iloc[:, -1].astype(int)
    return df

def design_matrix(feat):
    out = np.zeros((feat.shape[0], feat.shape[1] + 1))
#     print(out)
    for i in range(len(feat)):
        out[i] = np.array([1] + list(feat[i]))
    return out


# # Q_01

# In[975]:


def balanced_error(pred_y, label_y, report = True):

    fn = np.sum(np.logical_and((pred_y == 0), (label_y == 1)))
    tp = np.sum(np.logical_and((pred_y == 1), (label_y == 1)))

    fp = np.sum(np.logical_and((pred_y == 1), (label_y == 0)))
    tn = np.sum(np.logical_and((pred_y == 0), (label_y == 0)))

    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)

    ber =  1 - 0.5 * ((tp / (tp + fn)) + (tn / (tn + fp)))
#     ber_check = 0.5 * (fpr + fnr)
#     assert ber_check == ber, 'sth is wrong'

#     print(tp, fp, tn, fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    if report:
        print('accuracy: ', accuracy,'\n','BER: ', ber)
    return accuracy, ber

def train_and_predict(feat, lab, classifier, input_pred):
    '''return the trained clf;
    feat, lab and input_pred should have same shapes
    '''
    return classifier.fit(feat, lab).predict(input_pred).reshape(-1, 1)


def train_validation_test_split(X, y, test_size = 0.25, val_size = 0.25, shuffled = True):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size = test_size, shuffle = shuffled)

    val_s = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size = val_s, shuffle = shuffled)

#     print(X_train.shape, X_val.shape, X_test.shape)

    return X_train, X_val, X_test, y_train, y_val, y_test


# In[976]:


# navie trainin
# In[422]:


# print('\ntraining set: ')
# balanced_error(train_pred, y_train);

# print('\nvalidation set: ')
# balanced_error(val_pred, y_val);

# print('\ntest set: ')
# balanced_error(test_pred, y_test);


# # Q_04

# ### I would select the one with the lowest BER on validation set. Since the validation set withholds part of the unseen data from training, it allows us to tune the model by selecting the one with the lowest error rate. In this case, I would choose the classifier when c = 0.01. I run mutiple times, since the data is shuffled and each time it is likely to get different answer. c= 0.01 is the most frequent one.

# In[980]:


def LogisticReg_pip(X, y, c_range = (10. ** np.arange(-4, 5, 1))):
    train_scores = []
    validation_scores = []
    test_scores = []

    X_train, X_val, X_test, y_train, y_val, y_test =    train_validation_test_split(X, y)

    for c in c_range:
#         X_train, X_val, X_test, y_train, y_val, y_test =\
#         train_validation_test_split(X, y)
        classifier = LogisticRegression(C = c, class_weight= 'balanced' )

        pred_y_train = train_and_predict(X_train, y_train, classifier, X_train)
        pred_y_val = train_and_predict(X_train, y_train, classifier, X_val)
        pred_y_test = train_and_predict(X_train, y_train, classifier, X_test)


#         accur, ber = balanced_error(pred_y, label_y)
        train_ber = balanced_error(pred_y_train, y_train, False)[1];
#         print(pred_y_train.shape, y_train.shape)
        val_ber = balanced_error(pred_y_val, y_val, False)[1];
        test_ber = balanced_error(pred_y_test, y_test, False)[1];

        train_scores.append(train_ber)
        validation_scores.append(val_ber)
        test_scores.append(test_ber)

#     print(train_scores)
    return pd.DataFrame(data = {'train_scores': train_scores,
                                  'validation_scores': validation_scores,
                                  'test_scores': test_scores},
                       index = c_range)




def F_score(pred_y, label_y, beta = 1):


    fn = np.sum(np.logical_and((pred_y == 0), (label_y == 1)))
    tp = np.sum(np.logical_and((pred_y == 1), (label_y == 1)))

    fp = np.sum(np.logical_and((pred_y == 1), (label_y == 0)))
    tn = np.sum(np.logical_and((pred_y == 0), (label_y == 0)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f_beta = (1 + beta ** 2) *             (precision * recall / (beta ** 2 * precision + recall))

#     f_beta_check = 1 - 0.5 * (tp / (tp + fn) + tn / (tn + fp))
#     print(f_beta, f_beta_check)
#     assert f_beta == f_beta_check, 'inconsistent F_beta'
    return f_beta


def output_result():
    print('xxxx')
    return



# In[919]:


# X_train, X_val, X_test, y_train, y_val, y_test = \
# train_validation_test_split(X, y, 0.25, 0.25, shuffled = True



# In[ ]:





# In[ ]:





# In[ ]:
