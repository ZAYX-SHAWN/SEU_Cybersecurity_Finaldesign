import os
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
from keras.models import Model,load_model,Sequential
from keras.applications.xception import  Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet_v2 import EfficientNetV2L
import keras.callbacks as kcallbacks
import keras
#from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import img_to_array
import math
import random
from keras.utils import plot_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#generate training and test images
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=32	#could try 128 or 32

#Normalization
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        r'D:\final\train_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        r'D:\final\test_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')

input_shape=INPUT_SIZE
num_class=15
epochs=15
savepath=r'D:\final\trained_models\VGG19.h5'
'''
#Define CNN Model
model = Sequential()
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=input_shape,padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(256,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_class,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.0076, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

#Plot Training and Testing Accuracies
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN Model Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid()
plt.show()

#Plot Training and Testing Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN Model Loss', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()
plt.show()
'''

# Define VGG19 Model
model_fine_tune = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:40]:
        layer.trainable = False
for layer in model_fine_tune.layers[40:]:
        layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model = Dense(units=128, activation='relu')(model)
model = Dropout(0.3)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='vgg')
opt = keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                           mode='auto')
history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator), callbacks=[saveBestModel])

# Plot Training and Testing Accuracies
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('VGG19 Model Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid()
plt.show()

# Plot Training and Testing Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG19 Model Loss', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()
plt.show()

