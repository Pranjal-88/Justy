import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pickle

#TODO:Data Extraction
dataset=pd.read_csv(r'Datsets\train_csp.csv')
# print(dataset.shape)
# (76020, 371)

#TODO:Subdivison into features and labels
data_fts=dataset.drop(['TARGET','ID'],axis=1)
data_lbs=dataset['TARGET']

#TODO:Data Divison
train_x,test_x,train_y,test_y=train_test_split(data_fts,data_lbs,random_state=42,test_size=0.2)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (60816, 369)
# (60816,)    
# (15204, 369)
# (15204,)    

#TODO:Removing constt,quasi constt and duplicate columns
vt_ins=VarianceThreshold(threshold=0.01)
train_x=vt_ins.fit_transform(train_x)
test_x=vt_ins.transform(test_x)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (60816, 271)
# (60816,)    
# (15204, 271)
# (15204,)  

#TODO:Removing duplicate columns
train_x_T=train_x.T
test_x_T=test_x.T
train_x_T=pd.DataFrame(train_x_T)
dup=train_x_T.duplicated()
drop_index=[not index for index in dup]
train_x=train_x_T[drop_index].T
test_x=test_x_T[drop_index].T
# print(train_x.shape)
# print(test_x.shape)
# (60816, 254)
# (15204, 254)

#TODO:Standardisation
ins=StandardScaler()
train_x=ins.fit_transform(train_x)
test_x=ins.transform(test_x)

#TODO:Reshaping (ig CNN layers work only on 2D :( )
train_x=train_x.reshape(60816, 254,1)
test_x=test_x.reshape(15204, 254,1)

#TODO:Model construction
# model=keras.models.Sequential()

# model.add(keras.layers.Conv1D(filters=32,activation='relu',kernel_size=3,padding='same'))
# model.add(keras.layers.MaxPool1D(padding='valid',pool_size=2,strides=2))
# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Conv1D(filters=64,activation='relu',kernel_size=3,padding='same',kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(keras.layers.MaxPool1D(padding='valid',pool_size=2,strides=2))

# model.add(keras.layers.Conv1D(filters=128,activation='relu',kernel_size=3,padding='same',kernel_regularizer=keras.regularizers.l2(0.01)))
# model.add(keras.layers.MaxPool1D(padding='valid',pool_size=2,strides=2))

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dropout(0.3))

# model.add(keras.layers.Dense(units=128,activation='relu'))
# model.add(keras.layers.Dense(units=1,activation='sigmoid'))

#TODO:Model compilation
# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#TODO:Model fitting
# x=model.fit(train_x,train_y,epochs=10,batch_size=128,validation_data=(test_x,test_y))
# with open('cust_Sat.pkl','wb') as f:
#     pickle.dump(x.history,f)

#TODO:Learning curves
with open('cust_Sat.pkl','rb') as f:
    x=pickle.load(f)
fig,(ax1,ax2)=plt.subplots(1,2)

ax1.plot(range(1,11),x['accuracy'])
ax1.plot(range(1,11),x['val_accuracy'])
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend(['Training Data','Test Data'])

ax2.plot(range(1,11),x['loss'])
ax2.plot(range(1,11),x['val_loss'])
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend(['Training Data','Test Data'])

plt.show()