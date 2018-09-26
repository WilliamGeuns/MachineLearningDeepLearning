# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 00:46:19 2018

@author: William
"""

import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

# Loading data from preprocessing
pickle_in = open("X1.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y1.pickle","rb")
y = pickle.load(pickle_in)

# Give it unique name for tensorboard and also save
NAME = "Algorithm test2".format(int(time.time()))

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)

model = Sequential()
     
model.add(Conv2D(512, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
     
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
    
model.add(Flatten()) 
    
# Last dense layers must (not sure) have number of labels in data in parenthesis
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.2)) 
    
model.add(Dense(2))
model.add(Activation('softmax'))
    
model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
# Visualizing model open cmd cd to folder where the script is saved
# and type "tensorboard --logdir=logs\"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
     
model.fit(X, y, 
          batch_size=10,
          epochs=8, 
          validation_split=0.2,
          callbacks=[tensorboard]
          )
model.save(NAME)


    
    



