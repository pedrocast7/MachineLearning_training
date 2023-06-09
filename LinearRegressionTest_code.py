##python code for load and visualize a Linear regression type of dataset 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

##Generates sample data
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4) 
##Split data in train/test samples/labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) 

##Plotting the dataset
fig = plt.figure(figsize=(8,6))
plt.scatter( X[:, 0], y, color = 'b', marker = 'o', s = 30)
plt.show()


##Showing data properties
print(f'The shape of X is {X_train.shape}.')
print(f'The shape of y is {y_train.shape}.')


##Testing the Linear Regression implementation
from ML_algorithms_scratch_implementation import LinearRegression

##Creates a regressor instance from LR class
regressor = LinearRegression(lr=0.01)
##Train the model
regressor.fit(X_train, y_train)
##Test the model
predicted_values = regressor.predict(X_test)

##Define the Mean Square Error for evaluate the model
def MSE(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse_value = MSE(y_test, predicted_values)

print(f'The MSE of the model is about {mse_value}.')