import os
import numpy as np

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

# testing data
X_t = test