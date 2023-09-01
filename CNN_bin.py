import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

#TODO:Data Reading
dataset=pd.read_csv('Datsets\Pro2.csv')

#TODO:Columns
# print(dataset.columns)
['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',  
'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
'IsActiveMember', 'EstimatedSalary', 'Exited']

#TODO:Dropping irrelevant columns
dataset.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1,inplace=True)

#TODO:Hot Encoding
dataset=pd.get_dummies(dataset,drop_first=True,dtype=int)
# print(dataset.head())

#TODO:Data divison
data_fts=dataset.drop('Exited',axis=1)
data_lbs=dataset['Exited']
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,test_size=0.2,random_state=42)
# print(train_x.shape)    
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (8000, 11)
# (8000,)   
# (2000, 11)
# (2000,) 

#TODO:Normalisation
ins=StandardScaler()
train_x=ins.fit_transform(train_x)
test_x=ins.transform(test_x)

#TODO:Model build
model=keras.models.Sequential()

#TODO:Adding first layer
model.add(keras.layers.Dense(units=6,activation='relu',input_dim=11))

#TODO:Adding second layer
model.add(keras.layers.Dense(units=6,activation='relu'))

#TODO:Output layer
model.add(keras.layers.Dense(units=1,activation='sigmoid'))

#TODO:Model compilation
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# print(model.summary())

#TODO:Model fitting
model.fit(train_x,train_y,batch_size=10,epochs=20)

#TODO:Model evaluation
test_std,test_acc=model.evaluate(test_x,test_y)

#TODO:Stats
# print(f"Test std:{test_std}\nTest Accuracy:{test_acc}")
# Test std:0.34052911400794983
# Test Accuracy:0.8619999885559082