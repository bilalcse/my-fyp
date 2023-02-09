from cProfile import label
from email.mime import image
from tkinter import image_names
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt


image_directory='datasets/'
no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]
INPUT_SIZE=64
#print(no_tumor_images)
#path='no0.jpg'
#print(path.split('.')[1])
for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape=(n,image_width,image_height,n_channels)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)
# Now we will normalized our data for training purpose

x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)


# Model Building
# 64,64,3
model=Sequential()

model.add(Conv2D(32,(3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history5=model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=20,validation_data=(x_test,y_test),shuffle=False)
model.save('BrainTumor20Epohs.h5')
model.summary()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(10,9))
plt.plot(history5.history["accuracy"])
plt.plot(history5.history["loss"] ,c="green")
plt.legend(["Accuracy", "Loss"])
plt.title("Accuracy vs loss")
plt.title('Accuracy on Model 64 Filtters - 3x3 kernel -  Conv Layers and 2 FC less unit, Batch 16')
plt.show()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(10,9))
plt.plot(history5.history["loss"])
plt.plot(history5.history["val_loss"] ,c="Green")
plt.title("loss vs val_loss")
plt.legend(["train loss", "Val_loss"])
plt.title('Loss on Model  64 Filtters - 3x3 kernel - 3 Conv Layers and 2 FC unit , Batch 16')
plt.show()
import matplotlib.pyplot as plt
fig, axs = plt.subplots(figsize=(10,9))
plt.plot(history5.history["accuracy"])
plt.plot(history5.history["val_accuracy"] ,c="Green")
plt.title("accuracy vs val_accuracy")
plt.legend(["train accuracy", "Val_accuracy "])
plt.title('Accuracy on Model 64 Filtters - 3x3 kernel - 3 Conv Layers and FC unit, Batch 16')
plt.show()
