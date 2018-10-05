# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 03:17:35 2018

@author: William
"""
import pickle 
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import load_model

CATEGORIES = ['Animal','Distractor']

# Immediatly testing the model = making predictions on unseen data
# Just a csv file that isn't included in the training set (Testing dataset)
# This is where we start using our test set so we load the other two pickles
pickle_in = open("X_test_1.pickle","rb")
X_test = pickle.load(pickle_in)
pickle_in = open("y_test_1.pickle","rb")
y_test = pickle.load(pickle_in)

# Loading the trained model for testing
model = load_model("ClusterPC1")

# Letting the above trained model predict on unseen data 
prediction = model.predict(X_test, batch_size=10, verbose=0)
# Visualizing these predictions
for i in prediction:
    print(i)

# Rounding the prediction to the classes
rounded_prediction = model.predict_classes(X_test, batch_size=10, verbose=0)
# Visualizing the best model should give you 11 animals and 14 distractors
for i in rounded_prediction:
    print(i)

# Confusion matrix to visualize how the model was performing
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, rounded_prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CATEGORIES,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CATEGORIES, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

# =============================================================================
# # Just checking if the model that we are loading is the same as that I was training
# # just a double check :)
# model.summary()
# model.get_weights()
# =============================================================================
