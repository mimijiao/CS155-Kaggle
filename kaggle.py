import os
import numpy as np
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

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
max_depth_accuracy = 0
max_leaf_accuracy = 0
kf = cross_validation.KFold(len(X), n_folds=5)
for depth in range(1, 5):
    for estimator in range(50, 500, 50):
        err = 0 
        for train_idx, test_idx in kf:
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=estimator)
            clf = clf.fit(X_train, Y_train)
            err += clf.score(X_test, Y_test)
        err /= 5.0
        if err > max_depth_accuracy:
            print err
            print depth
            print estimator
            max_depth_accuracy = err
            max_depth = depth
            best_est = estimator
print (max_depth_accuracy, max_depth, best_est)

for leaf_size in range(1, 20):
    err = 0 
    for train_idx, test_idx in kf:
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf=leaf_size))
        clf = clf.fit(X_train, Y_train)
        err += clf.score(X_test, Y_test)
    err /= 5.0
    if err > max_leaf_accuracy:
        max_leaf_accuracy = err
        min_leaf = leaf_size

print (max_leaf_accuracy, min_leaf)

# testing data
X_t = test

clf = clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=best_est)
clf = clf.fit(X, Y)

Y_t = clf.predict(X_t)
identifier = range(1, len(X_t)+1)

result = np.reshape(np.concatenate((identifier, Y_t)), (len(X_t), 2), order='F')
np.savetxt(directory+"\\kaggle_result.txt", result, fmt='%d,%d', delimiter=',', newline='\n')