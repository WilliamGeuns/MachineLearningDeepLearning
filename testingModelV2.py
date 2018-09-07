# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 03:17:35 2018

@author: William
"""

import numpy as np
import tensorflow as tf

CATEGORIES = ['Pen','Soccer ball','Purple','Red','Seven','Two']

def prepare(filepath):
    epoch_array = np.genfromtxt(filepath,delimiter=',')
    return epoch_array.reshape(-1,60,2500,1)

model = tf.keras.models.load_model("Testsave7-CNN.model")

# You can test the model here, change the path name to where you saved testing files and change foldername (STIM'#')
# And change the name of the csv file accordingly Neele 'Stimuli' #EPOCH , example below
# I think we need a lot more nodes in the Network == Research
prediction = model.predict([prepare('D:\\Datasets\\RecordedEEGdata\\Testing Epochs\\Purple\\Neele 7 3.csv')])
max_index = np.argmax(prediction)
print(prediction) # will be a list in a list.
print(max_index)
print(CATEGORIES[max_index])