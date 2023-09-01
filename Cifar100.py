import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
import pickle
from keras.models import load_model
from PIL import Image
import urllib

#TODO:Reference
ref={
    0: 'apple',             1: 'aquarium_fish',     2: 'baby',
    3: 'bear',              4: 'beaver',            5: 'bed',
    6: 'bee',               7: 'beetle',            8: 'bicycle',
    9: 'bottle',           10: 'bowl',             11: 'boy',
   12: 'bridge',           13: 'bus',              14: 'butterfly',
   15: 'camel',            16: 'can',              17: 'castle',
   18: 'caterpillar',      19: 'cattle',           20: 'chair',
   21: 'chimpanzee',       22: 'clock',            23: 'cloud',
   24: 'cockroach',        25: 'couch',            26: 'crab',
   27: 'crocodile',        28: 'cup',              29: 'dinosaur',
   30: 'dolphin',          31: 'elephant',         32: 'flatfish',
   33: 'forest',           34: 'fox',              35: 'girl',
   36: 'hamster',          37: 'house',            38: 'kangaroo',
   39: 'keyboard',         40: 'lamp',             41: 'lawn_mower',
   42: 'leopard',          43: 'lion',             44: 'lizard',
   45: 'lobster',          46: 'man',              47: 'maple_tree',
   48: 'motorcycle',       49: 'mountain',         50: 'mouse',
   51: 'mushroom',         52: 'oak_tree',         53: 'orange',
   54: 'orchid',           55: 'otter',            56: 'palm_tree',
   57: 'pear',             58: 'pickup_truck',     59: 'pine_tree',
   60: 'plain',            61: 'plate',            62: 'poppy',
   63: 'porcupine',        64: 'possum',           65: 'rabbit',
   66: 'raccoon',          67: 'ray',              68: 'road',
   69: 'rocket',           70: 'rose',             71: 'sea',
   72: 'seal',             73: 'shark',            74: 'shrew',
   75: 'skunk',            76: 'skyscraper',       77: 'snail',
   78: 'snake',            79: 'spider',           80: 'squirrel',
   81: 'streetcar',        82: 'sunflower',        83: 'sweet_pepper',
   84: 'table',            85: 'tank',             86: 'telephone',
   87: 'television',       88: 'tiger',            89: 'tractor',
   90: 'train',            91: 'trout',            92: 'tulip',
   93: 'turtle',           94: 'wardrobe',
   95: 'whale',            96: 'willow_tree',      97: 'wolf',
   98: 'woman',            99: 'worm'
}

#TODO:Data Extraction
# (train_x,train_y),(test_x,test_y)=keras.datasets.cifar100.load_data()

#TODO:Shaping
# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
(50000, 32, 32, 3)
(50000, 1)        
(10000, 32, 32, 3)
(10000, 1) 

#TODO:Getting the pixel range
# print(np.min(train_x),np.max(train_x))
# print(np.min(train_y),np.max(train_y))
# 0 255
# 0 99

#TODO:Normalization
# train_x=train_x/255
# test_x=test_x/255

#TODO:Model extraction
# model=keras.models.Sequential()

#TODO:Model building
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same',input_shape=[32, 32, 3]))
# model.add(keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))

# model.add(keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))

# model.add(keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu',padding='same'))
# model.add(keras.layers.MaxPool2D(strides=2,pool_size=2,padding='valid'))


# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(units=128,activation='relu'))
# model.add(keras.layers.Dense(units=100,activation='softmax'))

#TODO:Model compilation
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

#TODO:Model fitting
# x=model.fit(train_x,train_y,batch_size=128,epochs=20,validation_data=(test_x,test_y))
# model.save('Model_Cifar100.h5')
# with open('Cifar100.pkl','wb') as f:
#     pickle.dump(x.history,f)

#TODO:learning curve for accuracy
# with open('Pickles\Cifar100.pkl','rb') as f:
#     x=pickle.load(f)
# fig,(ax1,ax2)=plt.subplots(1,2)
# ax1.plot(range(1,21),x['sparse_categorical_accuracy'])
# ax1.plot(range(1,21),x['val_sparse_categorical_accuracy'])
# ax1.legend(['Training Set','Test Set'])
# ax1.set_xlabel('Epochs')
# ax1.set_ylabel('Accuracy')

# ax2.plot(range(1,21),x['loss'])
# ax2.plot(range(1,21),x['val_loss'])
# ax2.legend(['Training Set','Test Set'])
# ax2.set_xlabel('Epochs')
# ax2.set_ylabel('Loss')
# plt.show()

#TODO:Model loading
model=load_model('Models\Model_Cifar100.h5')

#TODO:Real model
img_pth=r'https://www.hondaindiapower.com/admin/public/uploads/Products/jlmezzzik_Cover.jpg'
img1=Image.open(urllib.request.urlopen(img_pth)).convert('RGB')
# img1=Image.open(img_pth).convert('RGB')
img=img1.resize((32,32))
img=np.array(img)
img=img/255
img=np.expand_dims(img,axis=0)

pred=model.predict(img)
pred=np.argmax(pred,axis=1)

plt.imshow(img1)
plt.title(f'Prediction:{ref[pred[0]].capitalize()}'.replace('_',' '),fontweight='bold')
plt.axis('off')
plt.show()