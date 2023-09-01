import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import pickle
from keras.models import load_model
from PIL import Image

#TODO:Dataset Extraction
(train_x,train_y),(test_x,test_y)=keras.datasets.mnist.load_data()

#TODO:Max and min values
# print(np.min(train_x),np.max(train_x))
# print(np.min(train_y),np.max(train_y))
# 0 255
# 0 9  

#TODO:Normalisation
# train_x=train_x/255
# test_x=test_x/255

#TODO:Reshaping the images
# train_x=train_x.reshape(-1,28,28,1)
# test_x=test_x.reshape(-1,28,28,1)

#TODO:Model construction
model=keras.models.Sequential()
# model.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[28,28,1]))
# model.add(keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(units=128,activation='relu'))
# model.add(keras.layers.Dense(units=10,activation='sigmoid'))

#TODO:Model compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
# print(model.summary())

#TODO:Model fitting
# x=model.fit(train_x,train_y,batch_size=128,epochs=20,validation_data=(test_x,test_y))
# with open('digit.pkl','wb') as f:
#     pickle.dump(x.history,f)

#TODO:Model saving:
# model.save('Model_digit.h5')

#TODO:Model loading
model=load_model('Model_digit.h5')

#TODO:Learning curve for accuracy
with open('digit.pkl','rb') as f:
    x=pickle.load(f)
plt.plot(range(1,21),x['sparse_categorical_accuracy'])
plt.plot(range(1,21),x['val_sparse_categorical_accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training data','Test data'])
plt.show()

#TODO:Learning curve for loss
# with open('digit.pkl','rb') as f:
#     x=pickle.load(f)
# plt.plot(range(1,21),x['loss'])
# plt.plot(range(1,21),x['val_loss'])
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend(['Training data','Test data'])
# plt.show()

#TODO:Fitting model on real images
# img_pth=r'Piks\8.jpg'
# img=Image.open(img_pth).convert('L')
# img=img.resize((28,28))
# img=np.array(img)
# img=img.reshape(28,28,1)
# img=img/255
# img=np.expand_dims(img,axis=0)

# pred=model.predict(img)
# pred=np.argmax(pred,axis=1)

# print(pred)

