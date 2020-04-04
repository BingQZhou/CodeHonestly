#!/usr/bin/env python
# coding: utf-8

# ### Kaggle user name: Couson

# In[27]:


import gzip
from collections import defaultdict
import numpy as np


# In[28]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')

def parse_data_file(path):
    with open(path) as f:
        for l in f:
            yield l.strip().split(',')
    


# In[29]:


### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)
for user,book,r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    userRatings[user].append(r)


# # Q_01
# 
# ### performance of baseline model on validation set: 0.64905
# 

# In[4]:


allUsers = []
allBooks = []
pairs = []
books_per_user = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
    allUsers.append(user)
    allBooks.append(book)
    
    books_per_user[user].append(book)
    pairs.append(user + '-' + book)

uniqueUsers = list(set(allUsers))
uniqueBooks = list(set(allBooks))
# pairs = [allUsers[i] + '-' + allBooks[i] for i in ]
# len(pairs)


# In[5]:


# loop_break = 0
# not_read = []
    
# while (loop_break < 10000):

#     rand_user = uniqueUsers[np.random.choice(np.arange(len(uniqueUsers)))]
#     rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
#     rand_pair = rand_user + '-' + rand_book

#     if (rand_pair not in pairs):
#         not_read.append(rand_pair)
#         loop_break += 1
# not_read


# books_per_user

### re-built training data sets w/ 0/1 version 1
# loop_break = 0
# with open("train_Interactions.csv", 'w') as val:
    
    
#     for user,book,r in readCSV("train_Interactions.csv.gz"):
#         to_write = user + ',' + book + ',1\n'
#         val.write(to_write)
        
#         if loop_break >= 190000:
#             rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
#             while (rand_book in books_per_user[user]):
#                 rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
#             to_write = user + ',' + rand_book + ',0\n' 
#             val.write(to_write)

#         loop_break += 1    
            
        
        
# #     for pair in not_read:
# #         to_write = pair.replace('-', ',') + ',0\n'
# #         val.write(to_write)
# val.close()
# count


# In[6]:


### re-built training data sets w/ 0/1 version 2
loop_break = 0
with open("train_Interactions_A01.csv", 'w') as val:
    for user,book,r in readCSV("train_Interactions.csv.gz"):
        to_write = user + ',' + book + ',1\n'
        val.write(to_write)
        
        rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
        while (rand_book in books_per_user[user]):
            rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
        to_write = user + ',' + rand_book + ',0\n' 
        val.write(to_write)

        loop_break += 1    
            
        
        
#     for pair in not_read:
#         to_write = pair.replace('-', ',') + ',0\n'
#         val.write(to_write)
val.close()
loop_break


# In[7]:


### train validation data set split
val_data = open("read_validation.txt", 'w')
train_data = open("read_train.txt", 'w')

THRESHOLD = 0
with open("train_Interactions.csv", 'r') as all_data:
    for l in all_data:
        if THRESHOLD < 190000:
            train_data.write(l)
        else:
            val_data.write(l)
        
        THRESHOLD += 1

all_data.close()
val_data.close()
train_data.close()


# In[8]:


c1 = 0
training_data = parse_data_file('read_train.txt')
validation_data = parse_data_file('read_validation.txt')
for i,j,k in validation_data:
    c1 += 1
    
# print(c1)
assert (c1 == 190000) or (c1 == 20000), 'check data set num_rows!'


# In[9]:


### Would-read baseline: just rank which books are popular
### and which are not, and return '1' if a book is among 
### the top-ranked
training_data = parse_data_file('read_train.txt')
validation_data = parse_data_file('read_validation.txt')

bookCount = defaultdict(int)
totalRead = 0

for user, book, _ in training_data: #readCSV("train_Interactions.csv.gz")
#     print(user, book, _)
    bookCount[book] += 1
    totalRead += 1
# print(totalRead)
mostPopular = [(bookCount[x], x) for x in bookCount]
# print(mostPopular)
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/1.72: break
# print(return1)

# return1

predictions = open("read_predictions.txt", 'w')
for l in open("read_validation.txt"):
    if l.startswith("userID"):
        #header
        predictions.write(l)
        continue
#     print(l.strip().split(','))
    u,b,r = l.strip().split(',')
    if b in return1:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")

predictions.close()


# # Q_02
# ### adjusted precentile : 1/1.72 ~= 58% with a higher accuracy of  validation_data: 0.65325

# In[10]:


### eval the baseline model w/ validation set
def files_eval(file1, file2, format_ = 'txt'):
    assert (format_ == 'txt'), 'only valid in txt'
    file1_parsed = parse_data_file(file1)
    file2_parsed = parse_data_file(file2)
    
    out = []
    for i, j in zip(file1_parsed, file2_parsed):
#         print(i, j)
        out.append(i == j)
    return np.mean(out)

files_eval('read_predictions.txt', 'read_validation.txt')


# # Q_03
# 
# ### jaccard similariy model with accuracy of 0.65115

# In[11]:


def jaccard(set_1, set_2):
    out = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
    return out

# def most_similar_with(bid, book_dict):
#     similarities = []
#     book_i = item_dict[bid]
#     for bid_2 in book_dict.keys():
#         if bid_2 == bid:
#             continue;
#         similarity = jaccard(book_i, book_dict[bid_2])
#         similarities.append([similarity, bid_2])
#     similarities.sort(reverse = True)
# #     print(similarity)
#     return similarities[0]
def most_similar_with(bid, book_dict):
    similarities = []
    book_i = item_dict[bid]
    for bid_2 in book_dict.keys():
        if bid_2 == bid:
            continue;
        similarity = jaccard(book_i, book_dict[bid_2])
#         print(similarity)
#     if similarities:
        similarities.append([similarity, bid_2])
#     else:
#         return [0, bid]
    
    similarities.sort(reverse = True)
    return similarities[0]

def get_subset_dict(set_of_bid, book_dict):
    out = defaultdict(set)
    for i in set_of_bid:
        out[i] = book_dict[i]
    return out


# In[12]:


JACCARD_THRES = 0.0115

training_data = parse_data_file('read_train.txt')
validation_data = parse_data_file('read_validation.txt')

user_dict = defaultdict(set)
item_dict = defaultdict(set)
for user, book, _ in training_data:
#     print(user, book, _)
    user_dict[user].add(book)
    item_dict[book].add(user)

# plot = []

predictions = open("read_predictions_jaccard.txt", 'w')
for l in open("read_validation.txt", 'r'):
#     if l.startswith("userID"):
#         #header
#         predictions.write(l)
#         continue
#     print(l.strip().split(','))
    uid,bid,_ = l.strip().split(',')
#     print(uid,bid,_ )
    books_read_by_uid = user_dict[uid]
    book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
#     plot.append([most_similar_with(bid, book_dict_subset)[0], ])


    pred = int(most_similar_with(bid, book_dict_subset)[0] > JACCARD_THRES)
#     print(pred)
#     break
    to_write = uid + ',' + bid + ',' + str(pred) + '\n'
#     print(to_write)
    predictions.write(to_write)

predictions.close()


# plot


# In[13]:


files_eval('read_predictions_jaccard.txt', 'read_validation.txt')


# # A_01
# 
# ### with jaccard similarity and baseline combined the accuracy could be improved to be 0.6715

# In[21]:


# train_Interactions_A01
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd


# In[22]:


### re-built training data sets w/ 0/1 version 2
# loop_break = 0
with open("train_Interactions_A01.csv", 'w') as df:
    for user,book,r in readCSV("train_Interactions.csv.gz"):
        to_write = user + ',' + book + ',1\n'
        df.write(to_write)
        
        rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
        while (rand_book in books_per_user[user]):
            rand_book = uniqueBooks[np.random.choice(np.arange(len(uniqueBooks)))]
        to_write = user + ',' + rand_book + ',0\n' 
        df.write(to_write)

#         loop_break += 1    
            
        
        
#     for pair in not_read:
#         to_write = pair.replace('-', ',') + ',0\n'
#         val.write(to_write)
val.close()
# loop_break


# In[23]:


new_user_dict = defaultdict(set)
new_item_dict = defaultdict(set)
for user, book, _ in parse_data_file('train_Interactions_A01.csv'):
#     print(user, book, _)
#     break
    new_user_dict[user].add(book)
    new_item_dict[book].add(user)


# In[24]:


# print(len(new_user_dict), len(new_item_dict))
percentile = defaultdict(int)
for idx in range(len(mostPopular)):
    item_c = mostPopular[idx][0]
    item = mostPopular[idx][1]
    percentile[item] = (totalRead - idx) / totalRead

def most_similar_with(bid, book_dict):
    similarities = []
    book_i = item_dict[bid]
    for bid_2 in book_dict.keys():
        if bid_2 == bid:
            similarities.append([0, 'fake_id'])
            continue;
        similarity = jaccard(book_i, book_dict[bid_2])
#         print(similarity)
#     print(similarities)
        similarities.append([similarity, bid_2])
    
    similarities.sort(reverse = True)
    return similarities[0]

def get_subset_dict(set_of_bid, book_dict):
    out = defaultdict(set)
    for i in set_of_bid:
        out[i] = book_dict[i]
    return out


# In[25]:


def pre_setting(percentile_cutoff = 1.72):
    training_data = parse_data_file('read_train.txt')
    # validation_data = parse_data_file('read_validation.txt')

    bookCount = defaultdict(int)
    totalRead = 0

    for user, book, _ in training_data: #readCSV("train_Interactions.csv.gz")
    #     print(user, book, _)
        bookCount[book] += 1
        totalRead += 1
    # print(totalRead)
    mostPopular = [(bookCount[x], x) for x in bookCount]
    # print(mostPopular)
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/percentile_cutoff: break
    return return1

def generate_new_feature():
    training_new = parse_data_file('data/read_train_features.txt')
    validation_new = parse_data_file('data/read_validation_features.txt')
    
    X_train_new = []
    y_train_new = []
    for i in training_new:
        X_train_new.append([float(num) for num in i[2:-1]])
        y_train_new.append(float(i[-1]))

    X_validation_new = []
    y_validation_new = []
    for i in validation_new:
        X_validation_new.append([float(num) for num in i[2:-1]])
        y_validation_new.append(float(i[-1]))
    return X_train_new, y_train_new, X_validation_new, y_validation_new

group_dict = defaultdict(int)
for i,j in zip(uniqueUsers, range(len(uniqueUsers))):
    group_dict[i] = j
#     break
# group_dict


# In[26]:


# adding features

# train_feat = open("data/read_train_features.txt", 'w')
# val_feat = open("data/read_validation_features.txt", 'w')

# loop_break_tr = 0 # 380000


for cutoff in np.arange(1.81, 1.812, 0.01):
    return1 = pre_setting(cutoff)
    train_feat = []
    val_feat = []
    train_lab = []
    val_lab = []


#     train_feat = open("data/read_train_features.txt", 'w')
#     val_feat = open("data/read_validation_features.txt", 'w')

    loop_break_tr = 0 # 380000
    #####
    for l in open('train_Interactions_A01.csv', 'r'):
        one_hot = [0] * len(uniqueUsers)
        uid,bid,read = l.strip().split(',')
        books_read_by_uid = user_dict[uid]
        book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)


        loop_break_tr += 1
        popularity = int(bid in return1) #percentile[bid] #
        jaccard_val = most_similar_with(bid, book_dict_subset)[0]
        one_hot[group_dict[uid]] = 1
        
#         print(np.sum(one_hot))
#         break
        
        feat = [jaccard_val] + [popularity]
#         print(feat)
#         break
#         new_feat = uid + ',' + bid + ',' + str(jaccard_val) + ',' + str(group) + ','+ read +'\n'
        if loop_break_tr <= 380000:
            train_feat.append([feat])
            train_lab.append([read])
        
#             train_feat.write(new_feat)
        else:
            val_feat.append([feat])
            val_lab.append([read])
#             val_feat.write(new_feat)
#     val_feat.close()
#     train_feat.close()
    #####

#     print(val_feat)
#     break
#     X_train_new, y_train_new, X_validation_new, y_validation_new = generate_new_feature()
#     for complexity in 10. ** np.arange(-1,0):
#     clf = LogisticRegression(C = complexity, fit_intercept = True)
    clf = MLPClassifier(learning_rate_init = 0.0001)
    clf.fit(train_feat, train_lab)
    print(cutoff, complexity, np.mean(clf.predict(val_feat) == val_lab))


# In[ ]:





# In[273]:


def calculate_feature(pair):
    uid, bid = pair.strip().split('-')
    books_read_by_uid = user_dict[uid]
    book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
    
    popularity = int(bid in return1) #percentile[bid] #
    jaccard_val = most_similar_with(bid, book_dict_subset)[0]
    group = group_dict[uid]
    out = [[popularity, jaccard_val, group]]
    return out


# In[282]:


with open('new_res/predictions_Read.txt', 'w') as submission:
    for l in open("pairs_Read.txt", 'r'):
        if l.startswith("userID"):
            #header
            submission.write(l)
            continue
    #     print(l)
        uid,bid = l.strip().split('-')

#         books_read_by_uid = user_dict[uid]
#         book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
        to_predict = calculate_feature(l)
        pred = int(clf.predict(to_predict)[0])
        to_write = uid + '-' + bid + ',' + str(pred) + '\n'
#         print(to_write)
#         break
        submission.write(to_write)
#         print(pred)
#         break
#         if pred:
#             to_write = uid + '-' + bid + ',1\n'
#         elif bid in return4:
#             to_write = uid + '-' + bid + ',1\n'
#         else:
#             to_write = uid + '-' + bid + ',0\n'


# training_new = parse_data_file('data/read_train_features.txt')
# validation_new = parse_data_file('data/read_validation_features.txt')

# def generate_new_feature():
#     training_new = parse_data_file('data/read_train_features.txt')
#     validation_new = parse_data_file('data/read_validation_features.txt')
    
#     X_train_new = []
#     y_train_new = []
#     for i in training_new:
#         X_train_new.append([float(num) for num in i[2:-1]])
#         y_train_new.append(float(i[-1]))

#     X_validation_new = []
#     y_validation_new = []
#     for i in validation_new:
#         X_validation_new.append([float(num) for num in i[2:-1]])
#         y_validation_new.append(float(i[-1]))
#     return X_train_new, y_train_new, X_validation_new, y_validation_new

# print(X_train_new[0], y_train_new[0])


# In[276]:


# # fitting model
# # X_train_new

# for cutoff in range(0., 2, 0.05):
#     return1 = pre_setting(percentile_cutoff)
#     X_train_new, y_train_new, X_validation_new, y_validation_new = generate_new_feature()
#     for complexity in 10. ** np.arange(-2, 5):
#         lg = LogisticRegression(C = complexity, fit_intercept = True)
#         lg.fit(X_train_new, y_train_new)
#         print(np.mean(lg.predict(X_validation_new) == y_validation_new))


# In[206]:





# In[ ]:





# In[18]:


JACCARD_THRES = 0.0115
POP_THRES = 3.315

training_data = parse_data_file('read_train.txt')
validation_data = parse_data_file('read_validation.txt')


bookCount = defaultdict(int)
totalRead = 0

for user, book, _ in training_data:
    bookCount[book] += 1
    totalRead += 1
    
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()
# print(mostPopular)

return4 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return4.add(i)
    if count > totalRead/POP_THRES:
        break
# print(return4)


# for l in open("read_train.txt", 'r'):
#     uid,bid,_ = l.strip().split(',')
#     books_read_by_uid = user_dict[uid]
#     book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
# #     print(most_similar_with(bid, book_dict_subset)[0])
# #     break
#     jaccard_val = most_similar_with(bid, book_dict_subset)[0]
# #     print(jaccard_val)
#     popularity = int(bid in return4)
    
#     to_write = uid + ',' + bid + ',' + str(popularity) + ',' + str(jaccard_val)
# #     print(to_write)
# #     break
#     train_feat.write(to_write)

# train_feat.close()

# # train_feat = open("read_train_features.txt", 'w')
# val_feat = open("read_validation_features.txt", 'w')

# for l in open("read_validation.txt", 'r'):
#     uid,bid,_ = l.strip().split(',')
    
#     books_read_by_uid = user_dict[uid]
#     book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
    
#     print(book_dict_subset)
#     break
#     jaccard_val = most_similar_with(bid, book_dict_subset)[0]
#     popularity = int(bid in return4)
    
#     to_write = uid + ',' + bid + ',' + str(popularity) + ',' + str(jaccard_val)
#     val_feat.write(to_write)
    
# val_feat.close()


# In[ ]:


predictions = open("read_predictions_combo.txt", 'w')
for l in open("read_validation.txt", 'r'):
    uid,bid,_ = l.strip().split(',')



files_eval('read_features.txt', 'read_validation.txt')


# # Q_05

# In[46]:


submission = open("res/predictions_Read.txt", 'w')
for l in open("pairs_Read.txt", 'r'):
    if l.startswith("userID"):
        #header
        submission.write(l)
        continue
#     print(l)
    uid,bid = l.strip().split('-')
    
    books_read_by_uid = user_dict[uid]
    book_dict_subset = get_subset_dict(books_read_by_uid, item_dict)
    pred = most_similar_with(bid, book_dict_subset)[0] > JACCARD_THRES
    
    if pred:
        to_write = uid + '-' + bid + ',1\n'
    elif bid in return4:
        to_write = uid + '-' + bid + ',1\n'
    else:
        to_write = uid + '-' + bid + ',0\n'
    
    submission.write(to_write)
submission.close()


# # Q_06
# 
# ### the top 10 most common words are:
# - [[4352759, 'the'],
# - [2310108, 'a'],
# - [2195423, 'to'],
# - [2143824, 'i'],
# - [1907205, 'of'],
# - [1476704, 'it'],
# - [1288074, 'is'],
# - [1251844, 'in'],
# - [1135441, 'thi'],
# - [1103153, 'that']]

# In[1]:


import string
from nltk.stem.porter import *


punc = string.punctuation
stemmer = PorterStemmer()


# In[2]:


training_data = []
validation_data = []
VAL_THRES = 190000

for chunk in readGz("train_Category.json.gz"):
    if VAL_THRES > 0:
        training_data.append(chunk)
    else:
        validation_data.append(chunk)
    
    VAL_THRES -= 1
    
print(len(training_data), len(validation_data))
    


# In[3]:


words_count = defaultdict(int)
total_words = 0

for chunk in training_data:
    review = chunk['review_text'].strip().lower()
    cs = [char for char in review if (char not in punc)]
    review = ''.join(cs)
    for word in review.split():
        word = stemmer.stem(word)
        words_count[word] += 1
        total_words += 1   
#     print(review)
#     break

print(total_words, len(words_count))


# In[33]:


### build most pop. 1000 word dict

# 30582758 442055
# 59629097
words_count_lst = [[words_count[w], w] for w in words_count]
words_count_lst.sort(reverse = True)

words_count_lst[:10]
# subset_word_lst = [w[1] for w in words_count_lst[:1000]]
# # len(subset_word_lst)
# # subset_word_lst[-10:]


# # Q_07
# 
# ### before improvement: 0.6671
# 
# # Q_08
# 
# ### tried dict_size of 20000 & C = 10 and improved to: 0.7024

# In[34]:


from sklearn.linear_model import LogisticRegression
DICT_SIZE = 19000


# In[ ]:


subset_word_lst = [w[1] for w in words_count_lst[:DICT_SIZE]]

word_id = dict(zip(subset_word_lst, range(len(subset_word_lst))))
word_set = set(subset_word_lst)


# In[ ]:


### generate features
def generate_feature(chunk):
#     group = group_dict['user_id']
    vec = [0] * DICT_SIZE
    review = chunk['review_text'].strip().lower()
    cs = [char for char in review if (char not in punc)]
    review = ''.join(cs)
    for word in review.split():
        word = stemmer.stem(word)
        if word in word_set:
            vec[word_id[word]] += 1
    return [1] + vec #+ [group]
#         word = stemmer.stem(word)
#         words_count[word] += 1
#         total_words += 1   


# In[ ]:


X_train = [generate_feature(d) for d in training_data]
y_train = [d['genreID'] for d in training_data]

X_val = [generate_feature(d) for d in validation_data]
y_val = [d['genreID'] for d in validation_data]


# In[ ]:


lg = LogisticRegression(C = 10, fit_intercept = False)
lg.fit(X_train, y_train)


# In[ ]:


pred = lg.predict(X_val)
np.mean(pred == np.array(y_val))


# In[34]:


#0.7414


# In[15]:


### Category prediction baseline: Just consider some of the most common words from each category

catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}

predictions = open("res/predictions_Category.txt", 'w')
predictions.write("userID-reviewID,prediction\n")
for l in readGz("test_Category.json.gz"):
#     print(generate_feat(l))
#     break
    cat = lg.predict([generate_feature(l)])[0]
    predictions.write(l['user_id'] + '-' + l['review_id'] + "," + str(cat) + "\n")

predictions.close()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




