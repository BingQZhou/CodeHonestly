import pandas as pd
import matplotlib.pyplot as plt

def clean(data):
    data = data.drop_duplicates()
    data.fillna(0,inplace=True)
    data = data[data['catagory']=='other'].copy()
    return data

def explore(data,print_ = True):
    if print_:
        print(data.info())
        print("mean price: "+str(data['price'].mean()))
        print("max price: "+str(data['price'].max()))
        print("quantity: "+str(len(data)))

def train(data, n=3):
    data = clean(data)
    x = data.drop(['price'])
    y = data['price']
    model = LogisticRegression(c = 1)
    model.train_n_folds(x,y,fold = n)
    return model

def validate(test_data,model):
    test_data = clean(test_data)
    x = test_data.drop(['price'])
    y = test_data['price']
    predicted = model.predict(x)
    accuracy = sum(predicted == y)/len(y)
    if accuracy<0.8:
        print('fail')
    else:
        print('success')
    return accuracy
