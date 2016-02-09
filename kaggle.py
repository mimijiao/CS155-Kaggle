import os
import numpy as np
from sklearn import tree, cross_validation

directory = os.path.dirname(os.path.abspath(__file__))
training_file = directory + '\\kaggle_training.txt'
testing_file = directory + '\\kaggle_testing.txt'
data = np.genfromtxt(training_file, delimiter = '|', dtype=str)
test = np.genfromtxt(testing_file, delimiter = '|', dtype=str)

labels = data[0, :1000]
data = data[1:].astype(int)
test = test[1:].astype(int)

# training data
X = data[:, :1000]
Y = data[:, 1000]

# cross validation
#min_depth_error = 5*len(X)
#min_leaf_error = 5*len(X)
#kf = cross_validation.KFold(len(X), n_folds=5)
#for depth in range(15, 55):
#    err = 0 
#    for train_idx, test_idx in kf:
#        X_train, X_test = X[train_idx], X[test_idx]
#        Y_train, Y_test = Y[train_idx], Y[test_idx]
#        clf = tree.DecisionTreeClassifier(max_depth=depth)
#        clf = clf.fit(X_train, Y_train)
#        Y_pred = clf.predict(X_test)
#        for i in range(len(X_test)):
#            if Y_pred[i] != Y_test[i]:
#                err += 1
#    if err < min_depth_error:
#        min_depth_error = err
#        min_depth = depth
#print (min_depth_error, min_depth)
#
#for leaf_size in range(1, 30):
#    err = 0 
#    for train_idx, test_idx in kf:
#        X_train, X_test = X[train_idx], X[test_idx]
#        Y_train, Y_test = Y[train_idx], Y[test_idx]
#        clf = tree.DecisionTreeClassifier(min_samples_leaf=leaf_size)
#        clf = clf.fit(X_train, Y_train)
#        Y_pred = clf.predict(X_test)
#        for i in range(len(X_test)):
#            if Y_pred[i] != Y_test[i]:
#                err += 1
#    if err < min_leaf_error:
#        min_leaf_error = err
#        min_leaf = leaf_size
#
#print (float(min_leaf_error)/len(X), min_leaf)

## testing data
X_t = test

clf = tree.DecisionTreeClassifier(min_samples_leaf=4)
clf = clf.fit(X, Y)

Y_t = clf.predict(X_t)
identifier = range(1, len(X_t)+1)

result = np.reshape(np.concatenate((identifier, Y_t)), (len(X_t), 2), order='F')
np.savetxt(directory+"\\kaggle_result.txt", result, fmt='%d,%d', delimiter=',', newline='\n')