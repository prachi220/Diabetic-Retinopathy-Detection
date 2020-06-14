# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import to_categorical
from PIL import Image
from PIL import ImageFile
import pandas as pd
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True
seed = 7
np.random.seed(seed)
imgX = 198
imgY = 297
input_shape = (imgX, imgY, 1)

csv_file = './trainLabels.csv'
df = pd.read_csv(csv_file)
mydict = dict(zip(df.image, df.level))
Y_new = []

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        if mydict.get(filename[0:-5]) is not None:
            img = load_img(os.path.join(folder, filename))
            Y_new.append(mydict[filename[0:-5]])
            img = img.resize((imgX, imgY),Image.ANTIALIAS)
            gray = img.convert('L')
            gray = img_to_array(gray)
            gray = gray.astype('float16')
            images.append(gray)
    return np.array(images, dtype='float16')

X = load_images('./train')
X = X.reshape(X.shape[0], imgX, imgY, 1)
X = X/255.0

Y = to_categorical(Y_new,num_classes=5)

idx = np.random.permutation(len(X))

print (idx, len(X), len(Y))
X1,Y1 = X[idx], Y[idx]

X_train = X1[:10000]
Y_train = Y1[:10000]
X_test = X1[10000:12000]
Y_test = Y1[10000:12000]
X_val = X1[12000:14000]
Y_val = Y1[12000:14000]

def createModel():
  print ("inside createModel") 
  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(5, activation='softmax'))
    
  return model

model1 = createModel()
batch_size = 100
epochs = 100
print ("model1 being compiled")
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print (len(X_train), len(Y_train), len(X_val), len(Y_val))
history = model1.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val))
print ("training completed")
print (history)
res = model1.evaluate(X_test, Y_test)
print ("evaluate completed")
print (res)

def perf_measure(y_actual, y_hat):
   TP = 0
   FP = 0
   TN = 0
   FN = 0

   for i in range(len(y_hat)): 
       if y_actual[i]>0 and y_hat[i]>0:
          TP += 1
       if y_hat[i]>0 and y_actual[i]==0:
          FP += 1
       if y_actual[i]==y_hat[i]==0:
          TN += 1
       if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
          FN += 1

   return(TP, FP, TN, FN)
   
pred = model1.predict(X_test)
Y_pred = np.argmax(pred, axis=-1)
Y_test = np.argmax(Y_test, axis=-1)   
(tp,fp,tn,fn) = perf_measure(Y_test, Y_pred)
print (tp,fp,tn,fn)

print(confusion_matrix(Y_test, Y_pred))
