import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import ResNet50

#->Data Extraction
gen=ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)

train_pth=r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\flowers\train'
test_pth=r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\flowers\test'
val_pth=r'C:\Users\HP\OneDrive\Desktop\PranCode\Datsets\flowers\valid'

train_data=gen.flow_from_directory(train_pth,batch_size=32,target_size=(224,224))
test_data=gen.flow_from_directory(test_pth,batch_size=32,target_size=(224,224))
val_data=gen.flow_from_directory(val_pth,batch_size=32,target_size=(224,224))

# print(train_data.class_indices)
# {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}

#->Modelling
base_model=ResNet50(include_top=False,weights='imagenet',input_shape=[224,224,3])

model=keras.models.Sequential()
model.add(base_model)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(5,activation='softmax'))

# print(model.summary())

#->Optimizing
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')

#->Training
model.fit(train_data,validation_data=val_data,epochs=5)
model.save('Models/Roserade.h5')