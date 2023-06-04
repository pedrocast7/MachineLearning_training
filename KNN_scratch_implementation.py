## Python code that implements the K-nearest neighbours algorithm 
import numpy as np
from collections import Counter


def euclidian_distance(x1, x2):
    ##global function that returns the euclidian distance for 2 points.
    return np.sqrt( np.sum((x1-x2)**2))

class KNN:

    '''
    Notation: 
    k = number of K numbers of closest neighbours, 3 is default
    X = vector of multiple samples
    x = Nth sample
    y = target vector
    '''

    def __init__(self, k=3): ## Takes the K numbers of closest neighbours, 3 is default
        self.k = k

    def fit (self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X] ## Takes the predicted labels for each sample.
        
        return np.array((predicted_labels))

    def _predict(self, x):
        ## compute euclidian distance 

        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]        

        ## get the K nearest samples and its labels
       
        k_indices = np.argsort(distances)[:self.k] ## Get the indices of the K nearest samples.
        k_nearest_labels = [self.y_train[i] for i in k_indices] ## Get the labels for the K nearest samples.

        ## majority vote to get most common class label
        most_common = Counter(k_nearest_labels).most_common(1) ## Gets only the most common class.
        return most_common[0][0] ## Returns only the class label value (the method above returns the times that the class ocurred as well).

