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
from tensorflow.keras.callbacks import TensorBoard

# Loading data from preprocessing
pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

# Give it unique name for tensorboard and also save
NAME = "Algorithm test7".format(int(time.time()))

# Assigning memory to not run OOM
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
session = tf.Session(config=config)

# Model starts here
model = Sequential()
 
model.add(Conv2D(4096, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
 
model.add(Conv2D(2048, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) 
 
# Last dense layers must (not sure) have number of labels in data in parenthesis
model.add(Dense(6))
model.add(Activation('softmax'))
 
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Visualizing model open cmd cd to folder where the script is saved
# and type "tensorboard --logdir=logs\"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
 
model.fit(X, y, 
          batch_size=5,
          epochs=10, 
          validation_split=0.1,
          callbacks=[tensorboard])
model.save("Testsave7-CNN.model")




# =============================================================================
# # Try different layers
# dense_layers = [0, 1, 2]
# layer_sizes = [32, 64, 128]
# conv_layers = [1, 2, 3]
# 
# for dense_layer in dense_layers:
#     for layer_size in layer_sizes:
#         for conv_layer in conv_layers:
#             NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
#             print(NAME)
# 
#             model = Sequential()
# 
#             model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
#             model.add(Activation('relu'))
#             model.add(MaxPooling2D(pool_size=(2, 2)))
# 
#             for l in range(conv_layer-1):
#                 model.add(Conv2D(layer_size, (3, 3)))
#                 model.add(Activation('relu'))
#                 model.add(MaxPooling2D(pool_size=(2, 2)))
# 
#             model.add(Flatten())
# 
#             for _ in range(dense_layer):
#                 model.add(Dense(layer_size))
#                 model.add(Activation('relu'))
# 
#             model.add(Dense(18))
#             model.add(Activation('softmax'))
# 
#             tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
# 
#             model.compile(loss='sparse_categorical_crossentropy',
#                           optimizer='adam',
#                           metrics=['accuracy'],
#                           )
# 
#             model.fit(X, y,
#                       batch_size=5,
#                       epochs=4,
#                       validation_split=0.3,
#                       callbacks=[tensorboard])
# =============================================================================
