import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


def main():

    # data = np.loadtxt(r'splice.data', delimiter=',', dtype='string')
    data = np.loadtxt(r'splice.data', delimiter=',', dtype=bytes).astype(str)

    bases = {'A': 0, 'C': 1, 'D': 2, 'G': 3, 'N': 4, 'R': 5, 'S': 6, 'T': 7}

    X_base = np.asarray([[bases[c] for c in seq.strip()] for seq in data[:, 2]])
    y_class = data[:, 0]

    enc = OneHotEncoder(n_values=[8]*X_base.shape[1])
    lb = LabelEncoder()

    enc.fit(X_base)
    lb.fit(y_class)

    X = enc.transform(X_base).toarray()
    y = lb.transform(y_class)

    sizes = [.2, .15, .1, .05, .01]
    algo_results = dict()

    for size in sizes:

        rs = ShuffleSplit(n_splits=2, test_size=size, random_state=0)
        # train_index, test_index = rs.split(X).next()
        train_index, test_index = rs.split(X).__next__()
        train_X, train_y = X[train_index], y[train_index]
        test_X, test_y = X[test_index], y[test_index]




def print_score(name, score):
    print('%s classifier got a score of %s' % (name, score))




from leven import levenshtein       
from sklearn.cluster import dbscan
data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
def lev_metric(x, y):
    i, j = int(x[0]), int(y[0])     # extract indices
    return levenshtein(data[i], data[j])

X = np.arange(len(data)).reshape(-1, 1)
dbscan(X, metric=lev_metric, eps=5, min_samples=2)
