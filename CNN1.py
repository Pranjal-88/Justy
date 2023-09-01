import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.metrics import confusion_matrix,accuracy_score
from PIL import Image

#TODO:Data Extraction
(train_x,train_y),(test_x,test_y)=keras.datasets.fashion_mnist.load_data()

#TODO:Shape check
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)

#TODO:Estimating pixel range
# print(np.max(train_x),np.min(train_x))
# (255,0)
# print(np.max(train_y),np.min(train_y))
# (9,0)

#TODO:Index meaning in y
ref={0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',
     6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}
 
#TODO:Image Display
# plt.imshow(train_x[1])
# plt.colorbar()
# plt.show()

#TODO:Standardiastion
# train_x=train_x/255
# test_x=test_x/255
# plt.imshow(train_x[115])
# plt.colorbar()
# plt.show()

#TODO:Flattening the datset
train_x=train_x.reshape(-1,28*28)
test_x=test_x.reshape(-1,28*28)

#TODO:Model building
#Sequence of layers
model=keras.models.Sequential()

#TODO:Adding first layer
# 1.units->number of neurons
# 2.activation function
# 3.input shape=784
model.add(keras.layers.Dense(units=128,activation='relu',input_shape=(784,)))

#TODO:Adding second layer with dropout
#regularization technique to prevent overfitting
model.add(keras.layers.Dropout(0.3))

#TODO:Output layer
model.add(keras.layers.Dense(units=10,activation='softmax'))

#TODO:Compliling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
# print(model.summary())

#TODO:Model training
model.fit(train_x,train_y,epochs=10)

#TODO:Model Evaluation
test_loss,test_accuracy=model.evaluate(test_x,test_y)

#TODO:Predictions
pred=model.predict(test_x)
pred = np.argmax(pred, axis=1)

#TODO:Accuracy and confusion matrix
acs=accuracy_score(test_y,pred)
cf=confusion_matrix(test_y,pred)
# print(acs)
# 0.7715

#TODO:Real image classification
img_pth=r'Piks\71Esqu3tVzL._SL1500_.jpg'
img=Image.open(img_pth).convert('L')
img1=img.resize((28,28))
img=np.array(img1)
img=img/255
img=img.reshape(1,28*28)
pred=model.predict(img)
pred=np.argmax(pred,axis=1)
print(pred)
plt.imshow(img1)
plt.show()