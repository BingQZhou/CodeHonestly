import pandas as pd
import matplotlib.pyplot as plt

def clean_data(data):
    data = data.drop_duplicates()
    data.dropna(inplace=True)
    data = data.where(data['catagory']=='other')
    return data

def explore_data(data):
    print(data.info())
    print("average price: "+str(data['price'].mean()))
    print("max price: "+str(data['price'].max()))
    print("quantity: "+str(len(data)))

def train_model(data):
    data = clean_data(data)
    x = data.drop(['price'])
    y = data['price']
    model = LogisticRegression()
    model.train_n_folds(x,y,fold = 5)
    return model

def validate_model(test_data,model):
    test_data = clean_data(test_data)
    x = test_data.drop(['price'])
    y = test_data['price']
    predicted = model.predict(x)
    accuracy = (predicted == y).mean()
    if accuracy >= 0.8:
        print('success')
    return accuracy
