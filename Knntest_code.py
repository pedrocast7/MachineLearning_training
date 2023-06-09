##python code for load and visualize the iris dataset properties 

import numpy as np
from sklearn import datasets ##importing datasets from sklearning library
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt ## library for plotting data
from matplotlib.colors import ListedColormap 

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF']) ## colormap to vary the plot in rgb color space

iris = datasets.load_iris() ## loading the famous iris dataset
X, y = iris.data, iris.target

## spliting data in train and test sub-sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234) 


## visualizing data properties
print(X_train.shape)
print(X_train[0])

print(y_train.shape)
#print(y_train)

##plotting data for visual understanding
plt.figure()
plt.title('X1 and X2 features for Iris dataset classes')
plt.xlabel('X1 feature')
plt.ylabel('X2 feature')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20) ##plotting only 2 of the 4 features for a 2D case (helps visualization)
plt.show()

from ML_algorithms_scratch_implementation import KNN ## imports the KNN class from the KNN_scratch_implementation (same folder)

clf = KNN(k=5)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

## calculates the accuracy 
acc = np.sum(predictions == y_test) / len(y_test)  ## the accuracy is defined by the number of right predictions divided by the number of total samples
print(f'The accuracy for this model is about {acc} %')