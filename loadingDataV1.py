# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 23:34:05 2018

@author: William
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random
import pickle

DATADIR = 'D:\Datasets\RecordedEEGdata\Epochs'
CATEGORIES = ['Pen','Soccer ball','Purple','Red','Seven','Two']

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    for epoch in os.listdir(path):
        epoch_array = np.genfromtxt(os.path.join(path,epoch),delimiter=',')
        plt.imshow(epoch_array)
        
        break
    break

print(epoch_array)
print(epoch_array.shape)

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for epoch in tqdm(os.listdir(path)):
            try:
                epoch_array = np.genfromtxt(os.path.join(path,epoch),delimiter=',')
               # epoch_array = pd.read_csv(os.path.join(path,epoch), names = np.arange(1,2501) )
                training_data.append([epoch_array, class_num])
            except Exception as e:
                pass
create_training_data()
print(len(training_data))

#            
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

#
X = []
y = []
for features,label in training_data:
        X.append(features)
        y.append(label)
X = np.array(X).reshape(-1,60,2500,1)

#print(X[0].reshape(-1, 60, 2500, 1))

# =============================================================================
#     feature = features.values()
#     output = neuro(feature.reshape(150000),label)
#     break
# =============================================================================
#y = np.array(y).reshape(-1,70,1,1)

# Saving the preprocessed data to feed in to algorithm 
# This way you dont have to extract all the csv files every time you change your algorithm
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


            
            
            
            
            