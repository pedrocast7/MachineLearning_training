### Python code that stores some of the most used Machine Learning algorithms as classes to just import
### and use.
### Author: Pedro Anderson Ferreira Castro // Year: 2023
 
import numpy as np
from collections import Counter


def euclidian_distance(x1, x2):
    ##global function that returns the euclidian distance for 2 points.
    return np.sqrt( np.sum((x1-x2)**2))

class KNN:
## Python code that implements the K-nearest neighbours algorithm
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




class LinearRegression:
## Python code that implements the K-nearest neighbours algorithm

    '''
    Notation:
    lr = learning rate parameter
    n_iterations = number of iterations for the training
    weights = weights vector
    bias = bias number
    X = vector of multiple samples
    y = target vector

    '''

    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_interations = n_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        ##Initialization of parameters
        n_samples, n_features = X.shape

        #Initialize the weight vector with 0 and also the bias.
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_interations):
            
            #Aproximation of the target
            y_predicted = np.dot(X, self.weights) + self.bias

            #Derivative of dW and dB
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            #Update the Weights and Bias values
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        

    def predict(self, X):
        #Predict using the model y_predict = W*x + b
        y_predicted = np.dot(X, self.weights) + self.bias
        
        return y_predicted

        