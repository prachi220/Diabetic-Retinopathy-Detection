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
# imgX = 396
# imgY = 594
imgX = 198
imgY = 297
input_shape = (imgX, imgY, 1)

csv_file = './trainLabels.csv'
df = pd.read_csv(csv_file)

# Separate majority and minority classes
df_majority = df[df.level==0]
df_minority = df[df.level>0]

# df_class_0_under = df_minority.sample(15000)
# df_upsampled = pd.concat([df_class_0_under, df_majority], axis=0)

# mydict = dict(zip(df_upsampled.image, df_upsampled.level))

# mydict = dict(zip(df.image, df.level))
# Downsample majority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample without replacement
                                 n_samples=15000,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])
 
# Display new class counts
# df_upsampled.balance.value_counts()
mydict = dict(zip(df_upsampled.image, df_upsampled.level))
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

# print (idx)
X1,Y1 = X[idx], Y[idx]

# pca = PCA(n_components=4096)
# X1 = pca.fit_transform(X1)
# X1 = X1.reshape(X1.shape[0], 64, 64, 1)
# new_input_shape = (64, 64, 1)

X_train = X1[:22000]
Y_train = Y1[:22000]
X_test = X1[22000:27000]
Y_test = Y1[22000:27000]
X_val = X1[27000:29000]
Y_val = Y1[27000:29000]

def createModel():
   model = Sequential()
   model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
   model.add(Conv2D(32, (3, 3), activation='relu'))
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
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val,Y_val))
res = model1.evaluate(X_test, Y_test)
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