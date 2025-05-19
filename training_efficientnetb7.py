from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
from keras.models import Model,load_model,Sequential
from keras.applications.efficientnet import EfficientNetB7
import keras.callbacks as kcallbacks
import keras
import matplotlib.pyplot as plt


#generate training and test images
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=32	#could try 128 or 32

#Normalization
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/autodl-fs/data/train_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        '/autodl-fs/data/test_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')

input_shape=INPUT_SIZE
num_class=15
epochs=15
savepath='/autodl-fs/trained_Models/EfficientNetB7.h5'

#Define EfficientNetB7 Model
model_fine_tune = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:350]:
    layer.trainable = False
for layer in model_fine_tune.layers[350:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=256,activation='relu')(model)
model=Dropout(0.2)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='efficientnetb7')
opt = keras.optimizers.Adam(learning_rate=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

#Plot Training and Testing Accuracies
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('EfficientNetB7 Model Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='lower right')
plt.grid()
plt.show()

#Plot Training and Testing Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('EfficientNetB7 Model Loss', fontsize=12, fontweight='bold')
plt.ylabel('Loss', fontsize=10, fontweight='bold')
plt.xlabel('Number of Epochs', fontsize=10, fontweight='bold')
plt.legend(['Train', 'Test'], loc='upper right')
plt.grid()