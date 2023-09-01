import os
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import urllib
from keras.models import load_model


#TODO:Deleting unknown extension files
# data_pth=r'C:\Users\HP\OneDrive\Desktop\PranCod
# for folders in os.listdir(data_pth):
#     folder_pth=data_pth+'\\'+folders
#     for images in os.listdir(folder_pth):
#         if not (images.endswith('.png') or images.endswith('.jpg') or images.endswith('.jpeg')):
#             os.remove(folder_pth+'\\'+images)

#TODO:Dataset formation
# labels=[]
# dataset=[]
# names=[]
# i=0
# data_pth=r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\PokemonData'
# for folders in os.listdir(data_pth):
#     folder_pth=data_pth+'\\'+folders
#     names.append(folders.replace('.png','').replace('.jpg','').replace('.jpeg',''))
#     for images in os.listdir(folder_pth):
#         labels.append(i)
#         img_pth=folder_pth+'\\'+images
#         img=Image.open(img_pth).convert('RGB')
#         img=img.resize((32,32))
#         img=np.array(img)
#         dataset.append(img)
#     i+=1
# np.save(r'Datsets\Pokemon_data.npy',dataset)
# with open(r'Pickles\Names.pkl','wb') as f:
#     pickle.dump(names,f)
# with open(r'Pickles\Pokemon_Labels.pkl','wb') as f:
#     pickle.dump(labels,f)

#TODO:Data Extraction
# train_data=np.load(r'Datsets\Pokemon_data.npy')
# print(train_data.shape)
# (6779, 32, 32, 3)
# with open(r'Pickles\Pokemon_Labels.pkl','rb') as f:
#     test_data=pickle.load(f)
# test_data=np.array(test_data)
# print(test_data.shape)
# (6779,)
# test_data=test_data.reshape(-1,1)
# print(test_data.max())
(6779, 1)

#TODO:Shuffling the dataset
# index=np.random.permutation(6779)
# train_data=train_data[index]
# test_data=test_data[index]

#TODO:Loading Names
with open(r'Pickles\Names.pkl','rb') as f:
    names=pickle.load(f)
    
#TODO:Image Display
# plt.imshow(train_data[24])
# plt.title(names[test_data[24][0]])
# plt.colorbar()
# plt.show()

#TODO:Data Divison
# train_x,test_x,train_y,test_y=train_test_split(train_data,test_data,random_state=42,test_size=0.2)
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# (5423, 32, 32, 3)
# (5423, 1)        
# (1356, 32, 32, 3)
# (1356, 1)  

#TODO:Normalisation
# train_x=train_x/255
# test_x=test_x/255

#TODO:Model Formation
# model=keras.models.Sequential()

#TODO:Model Layering
# model.add(keras.layers.Conv2D(filters=64,padding='same',kernel_size=3,activation='relu',input_shape=[32,32,3]))
# model.add(keras.layers.Conv2D(filters=64,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.MaxPool2D(padding='valid',pool_size=2,strides=2))

# model.add(keras.layers.Conv2D(filters=128,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.Conv2D(filters=128,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.MaxPool2D(padding='valid',pool_size=2,strides=2))

# model.add(keras.layers.Conv2D(filters=256,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.Conv2D(filters=256,padding='same',kernel_size=3,activation='relu'))
# model.add(keras.layers.MaxPool2D(padding='valid',pool_size=2,strides=2))

# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dropout(0.4))

# model.add(keras.layers.Dense(activation='relu',units=128))
# model.add(keras.layers.Dense(activation='softmax',units=149))

#TODO:Model Compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#TODO:Model Fitting and history saving
# x=model.fit(train_x,train_y,batch_size=128,validation_data=(test_x,test_y),epochs=20)
# with open(r'Pickles\Pokemon_history.pkl','wb') as f:
#     pickle.dump(x.history,f)

#TODO:Model Saving
# model.save(r'Models\Pokemon.h5')

#TODO:Learning Curves
# with open(r'Pickles\Pokemon_history.pkl','rb') as f:
#     x=pickle.load(f)
# fig,(ax1,ax2)=plt.subplots(1,2)

# ax1.plot(range(1,21),x['sparse_categorical_accuracy'])
# ax1.plot(range(1,21),x['val_sparse_categorical_accuracy'])
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Accuracy')
# ax1.legend(['Training Set','Test Set'])

# ax2.plot(range(1,21),x['loss'])
# ax2.plot(range(1,21),x['val_loss'])
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Loss')
# ax2.legend(['Training Set','Test Set'])

# plt.show()

#TODO:Model Extraction
model=load_model(r'Models\Pokemon.h5')

#TODO:Making real predictions
img_pth=r'https://staticg.sportskeeda.com/editor/2021/06/4341d-16230515794279-800.jpg'
# img1=Image.open(img_pth).convert('RGB')
img1=Image.open(urllib.request.urlopen(img_pth)).convert('RGB')
img=img1.resize((32,32))
img=np.array(img)
img=img/255
img=np.expand_dims(img,axis=0)

pred=model.predict(img)
pred=np.argmax(pred,axis=1)

plt.imshow(img1)
plt.title(f'Prediction:{names[pred[0]].capitalize()}',fontweight='bold')
plt.axis('off')
plt.show()






    
    