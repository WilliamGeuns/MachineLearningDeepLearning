# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 22:10:52 2018

@author: William
"""

import time
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard , EarlyStopping

#from sklearn.metrics import confusion_matrix
#import itertools
#import matplotlib.pyplot as plt


#CATEGORIES = ['Animals','Distractor']

# Loading training data from preprocessing
pickle_in = open("X_train_1.pickle","rb")
X = pickle.load(pickle_in)
pickle_in = open("y_train_1.pickle","rb")
y = pickle.load(pickle_in)

# Loading testing data for KFold algorithm
pickle_in = open("X_test_1.pickle","rb")
test_X = pickle.load(pickle_in)
pickle_in = open("y_test_1.pickle","rb")
test_y = pickle.load(pickle_in)

# KFold 
kfold_splits=2
skf = StratifiedKFold(n_splits=kfold_splits, shuffle=True)

y = np.transpose(y)


def create_model():
    # Give it unique name for tensorboard and also save
    
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
    model.add(Dropout(0.2)) 
        
    model.add(Flatten()) 
        
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.3)) 
    
    # Last dense layers must have number of classes in data in the parenthesis
    # Also must be softmax
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model

def train_model(model,xtrain,ytrain,xval,yval):
    
    NAME = "Algorithm test13379337".format(int(time.time()))  
    # Visualizing model open cmd cd to folder where the script is saved
    # and type "tensorboard --logdir=logs\"
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            
    # Preventing overfitting through 'earlystopping' it will monitor val_loss and stop
    # computing when val_loss goes up even though there are more epochs
    earlystopping = EarlyStopping(monitor= 'val_loss',
                                  min_delta = 0, 
                                  patience= 2, 
                                  verbose = 0, 
                                  mode ='auto'
                                  )
    
    accuracy= model.fit(xtrain,ytrain,
              epochs = 20, 
              validation_data= (xval,yval),
              batch_size=10,
              callbacks=[tensorboard, earlystopping],
              shuffle=True
              )    

    return accuracy.History()

total_acc=[]
for index, (train_indices, val_indices) in enumerate(skf.split(X,y)):
    print("Training on fold:" + str(index+1)+"/{}".format(kfold_splits))
    
    #Generate batches
    xtrain, xval = X[train_indices], X[val_indices]
    ytrain, yval = y[train_indices], y[val_indices]

    # Clear model, and create it
    model = None
    model = create_model()

    # Debug message I guess
    print ("Training new iteration on " + str(xtrain.shape[0]) + " training samples, " 
    + str(xval.shape[0]) + " validation samples, this may be a while...")
    
    
    history = train_model(model, xtrain, ytrain, xval, yval)
    accuracy_history = history.history['acc']
    val_accuracy_history = history.history['val_acc']
    print ("Last training accuracy: " + str(accuracy_history[-1]) 
    + ", last validation accuracy: " + str(val_accuracy_history[-1]))

print("%.2f%% (+/- %.2f%%)" % (np.mean(total_acc*100), np.std(total_acc*100)))