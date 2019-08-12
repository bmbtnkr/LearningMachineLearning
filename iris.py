import sklearn
import numpy as np
import pandas as pd
# import pylab
from sklearn.datasets import load_iris


# ----------------------------------------------------------------------------------------------------
# Supervised Learning
# ----------------------------------------------------------------------------------------------------
"""
150 image samples of flowers with 4 features:
    sepal length (cm)
    sepal width (cm)
    petal length (cm)
    petal width (cm)

3 target classes:
    setosa
    versicolour
    virginica

"""
iris = load_iris()

# print(iris.DESCR)

X = iris.data
y = iris.target

# print(x.shape)
# print(y.shape)
# print(y)
# print(X[0])
# print(iris.target_names)

# print(dir(iris))

# display the points in a scatter plot
# import pylab
# pylab.scatter(x[:,0], x[:,1], c=y)
# pylab.plot()
# pylab.show()

# data_frame = pd.DataFrame(X)
# print(data_frame.head())

# import a Linear Classifier Support Vector Machine
from sklearn.svm import LinearSVC

# instantiate the classifier
clf = LinearSVC()

"""
Most ML algorithms contain a method called fit(), fit_transform(), and/or transform()

First the data has be converted into feature vectors (images -> vector arrays representing pixel intensity
values at each pixel; or string classifiers into an array of numbers [0, 1, 2, 3])

The training data has to be centered (get a z-score by converting it to a standard score; subtract the mean 
from x and divide by the standard deviation for index in x array)
This gives a normal distribution plot where the mean is centered at the top of an evenly distributed
bell curve; +/- 1 sigma = 68% variance of data from the mean, 2 sigma = 95%, 3 sigma = 99.7% 
x_prime = (x - mean) / std_dev

Then you apply the same transformation to the testing set or to newly obtained training examples before
the forecast. But you have to use the same mean and standard deviation values.

fit() calculates the parameters it needs to use (mean and standard deviation, or whatever the ML 
algorithm class needs) and saves them as internal objects

transform() applies the transformation to a particular set of examples

fit_transform() combines the two steps use for the internal fitting of parameters on the training set (x)
but also returns a transformed (x_prime).

Not all ML algorithm classes contain fit_transform;
"""

# train the classifier
clf.fit(X, y)

# get the 3 intercepts for the SVM
print(clf.intercept_)

# the coefficients used to do the classification later on
print(clf.coef_)

# create a new np array with data similar to X[0]
X_new = [[ 5.0,  3.6,  1.3,  0.25]]
X_new = np.asarray(X_new)
# print(X_new)

print(clf.predict(X_new))

from sklearn.linear_model import LogisticRegression
clf2 = LogisticRegression().fit(X, y)
result = clf2.predict_proba(X_new)

for i in result:
    for j in i:
        print(round(j, 4))

# ----------------------------------------------------------------------------------------------------
# Unsupervised Learning
# ----------------------------------------------------------------------------------------------------
"""
Dimensionality reduction: deriving a new set of artificial features that is smaller than the
original feature set while retaining most of the variance of the original data

4 dimensions -> 2 dimensions for projecting onto a scatter plot
Also useful for speeding up algorithms 1000+ dimensions -> smaller n dimensions

The most common technique for dimensionality reduction is Principal Component Analysis

PCA is done by using linear combinations of the original features using a truncated
Singular Value Decomposition of the matrix (x) so as to project data onto a base of the
top singular vectors

Look for coorelations and remove redundant data
remove featurs with coorelations that don't add useful information to the features set
"""

from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(X)
X_pca = pca.transform(X)

print(X.shape) # original data in 4 dimensions
print(X_pca.shape) # new data in 2 dimensions

import pylab
pylab.scatter(X_pca[:,0], X_pca[:,1], c=y) # lowercase y = original classifications
pylab.plot()
pylab.show()

"""
Another unsupervised method is clustering
"""
from sklearn.cluster import KMeans
from numpy.random import RandomState

rng = RandomState(42)


kmeans = KMeans(3, random_state=rng).fit(X_pca)
# print(kmeans.labels_)
# print(np.round(kmeans.cluster_centers_, decimals=2))

import pylab
pylab.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_) # lowercase y = original classifications
pylab.plot()
pylab.show()
