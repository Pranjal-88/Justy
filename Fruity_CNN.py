import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model

#->Image Data Generator
# gen=ImageDataGenerator(rescale=1./255)
# train_data=gen.flow_from_directory(directory=r'Datsets\Fruits\Training',target_size=(260,260),batch_size=32)
# print(train_data.class_indices)
lst=['Apple Golden 1', 'Avocado', 'Banana', 'Kiwi', 'Lemon', 'Mango', 'Raspberry', 'Strawberry']
# test_data=gen.flow_from_directory(directory=r'Datsets\Fruits\Testing',target_size=(260,260),batch_size=32,classes=train_data.class_indices)
# # print(test_data.class_indices)
# valid_data=gen.flow_from_directory(directory=r'Datsets\Fruits\Validation',target_size=(260,260),batch_size=32)

#->Modelling
model=keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),input_shape=[260,260,3]))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.Conv2D(filters=64,kernel_size=(3,3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.Conv2D(filters=128,kernel_size=(3,3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.Conv2D(filters=256,kernel_size=(3,3)))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=8,activation='softmax'))
print(model.summary())

#->Optimizing
# model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(train_data,validation_data=valid_data,epochs=2)
# model.save('Models\Fruits.h5')

#->Predictions
# model=load_model('Models\Fruits.h5')

# img1=Image.open(r'Datsets\Fruits\Testing\Kiwi\0_100.jpg').convert('RGB')
# img=img1.resize((260,260))
# img=np.array(img)
# img=img/255
# img=np.expand_dims(img,axis=0)

# pred=model.predict(img)
# pred=np.argmax(pred)

# plt.imshow(img1)
# plt.title(lst[pred])
# plt.show()


