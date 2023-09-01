import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from PIL import Image

#TODO:Data Extraction
(train_x,train_y),(test_x,test_y)=keras.datasets.cifar10.load_data()

#TODO:Shaping
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (50000, 32, 32, 3)
# (50000, 1)        
# (10000, 32, 32, 3)
# (10000, 1)  

#TODO:Computing boundary values
# print(np.min(train_x),np.max(train_x))
# 0 255
# print(np.max(train_y),np.min(test_y))
# 9 0

#TODO:Reference
ref={0:'Airplane', 1:'Automobile',2:'Bird',3:'Cat',4:'Deer',5:'Dog',6:'Frog',7:'Horse',8:'Ship',9:'Truck'}

#TODO:Normalization
# train_x=train_x/255
# test_x=test_x/255

#TODO:Model build
# model=keras.models.Sequential()

#TODO:Adding first convulational layer
# model.add(keras.layers.Conv2D(filters=32,padding='same',kernel_size=3,activation='relu',input_shape=[32, 32, 3]))

#TODO:Adding second convulational layer and maxpool layer
# model.add(keras.layers.Conv2D(filters=32,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))

#TODO:Adding third convulation layer
# model.add(keras.layers.Conv2D(filters=64,activation='relu',kernel_size=3,padding='same'))

#TODO:Adding fourth convulational layer and maxpool layer
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=2,strides=2,padding='valid'))

#TODO:Adding dropout layer
# model.add(keras.layers.Dropout(0.4))

#TODO:Adding the flattenig layer
# model.add(keras.layers.Flatten())

#TODO:Adding first dense layer
# model.add(keras.layers.Dense(units=128,activation='relu'))

#TODO:Adding output layer
# model.add(keras.layers.Dense(units=10,activation='softmax'))

#TODO:Model compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
# print(model.summary())

#TODO:Model fitting
# model.fit(train_x,train_y,batch_size=10,epochs=10)

#TODO:Getting accurateness on test data
# test_loss,test_acc=model.evaluate(test_x,test_y)
# print(f"Accureacy:{test_acc}\nLoss:{test_loss}")

#TODO:Model saving
# model.save('Cifar10_Model.h5')

#TODO:Model loading
model=load_model('Models\Cifar10_Model.h5')
# test_loss,test_acc=model.evaluate(test_x,test_y)
# print(f"Accureacy:{test_acc}\nLoss:{test_loss}")

#TODO:Real image identification
img_pth=r'Piks\Peacock.jpg'
img1=Image.open(img_pth).convert('RGB')
img=img1.resize((32,32))
img=np.array(img)
img=img/255
img=np.expand_dims(img,axis=0)

pred=model.predict(img)
print(pred)
pred=np.argmax(pred,axis=1)

plt.imshow(img1)
plt.title(f'Prediction:{ref[pred[0]]}')
plt.axis('off')
plt.show()



