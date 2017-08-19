from sklearn import tree, gaussian_process, neural_network, neighbors
import numpy as np
import matplotlib.pyplot as plt

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160, 60 ,38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
     [159,55,37], [171,75,42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

clf_tree = tree.DecisionTreeClassifier()
clf_gaussian = gaussian_process.GaussianProcessClassifier()
clf_neural = neural_network.MLPClassifier()
clf_neigbors = neighbors.KNeighborsClassifier()

clf_tree = clf_tree.fit(X,Y)
clf_gaussian = clf_gaussian.fit(X,Y)
clf_neural = clf_neural.fit(X,Y)
clf_neigbors = clf_neigbors.fit(X,Y)

prediction_tree = clf_tree.predict([[190,70,43]])
prediction_gaussian = clf_gaussian.predict([[190,70,43]])
prediction_neural = clf_neural.predict([[190,70,43]])
prediction_neighbors = clf_neigbors.predict([[190,70,43]])

print('Tree predicts: ', prediction_tree)
print('Gaussian predicts: ', prediction_gaussian)
print('Neural predicts: ', prediction_neural)
print('Neighbors predicts: ', prediction_neighbors)

