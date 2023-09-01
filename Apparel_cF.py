import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix
from PIL import Image
from keras.models import load_model

#TODO:Data Reading
(train_x,train_y),(test_x,test_y)=keras.datasets.fashion_mnist.load_data()

#TODO:Shapping
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (60000, 28, 28)
# (60000,)       
# (10000, 28, 28)
# (10000,)  

#TODO:Pixel boundary
# print(np.min(train_x),np.max(train_x))
# print(np.min(test_y),np.max(test_y))
# 0 255
# 0 9  

#TODO:Reference
ref={0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',
     6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}

#TODO:Standardisation
train_x=train_x/255
test_x=test_x/255

#TODO:Flattening
# train_x=train_x.reshape(-1,28*28)
# test_x=test_x.reshape(-1,28*28)

#TODO:Model build
# model=keras.models.Sequential()

#TODO:Adding first layer
# model.add(keras.layers.Dense(units=128,activation='relu',input_shape=(784,)))

#TODO:Trying to avoid overfitting
# model.add(keras.layers.Dropout(0.3))

#TODO:Adding final layer
# model.add(keras.layers.Dense(units=10,activation='softmax'))

#TODO:Model compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#TODO:Model fitting
# model.fit(train_x,train_y,epochs=10)
# print(model.summary())

#TODO:Model evaluation
# loss,acc=model.evaluate(test_x,test_y)
# print(loss,acc)

#TODO:Predictions
# pred_test=model.predict(test_x)
# pred_test=np.argmax(pred_test,axis=1)

#TODO:Confusion matrix and accuracy score
# print(accuracy_score(test_y,pred_test))
# print(confusion_matrix(test_y,pred_test))

#TODO:Saving the model
# model.save('Ap_CNN.h5')

#TODO:Model
model=load_model('Ap_CNN.h5')

#TODO:Dealing with real data
img_pth=r'Piks\b1da1_512.jpg'
img1=Image.open(img_pth)
img=img1.convert('L')
img=img.resize((28,28))
img=np.array(img)
img=img/255
img=img.reshape(1,28*28)
pred=model.predict(img)
pred=np.argmax(pred,axis=1)

plt.imshow(img1)
# plt.colorbar()
plt.title(f"Prediction:{ref[pred[0]]}")
plt.show()

