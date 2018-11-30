import numpy as np
import pandas as pd
import math as math
import utils as utils
import sys as sys

def prepare_X(parameters):
    X = parameters.select_dtypes(include=['float64'])
    X = X.dropna(axis=1, how='all')
    X = X.apply(utils.fill_missing_data)
    X = X.apply(utils.normalize)
    X = utils.add_bias(X)
    return X

def g(z):
    return 1. / (1. + np.exp(-z))

def cost(w, X, y):
    h = g(X.dot(w.T))
    return sum((-np.log(h) * y - np.log(1. - h) * ~y)) / len(y)

def partial_deriv(w, X, y, j):
    h = g(X.dot(w.T))
    return sum(X[X.columns[j]] * (h - y)) / len(y)

def iterate_one_vs_all(X, y, w, alpha=0.3):
    w = w.copy()
    N = len(X.columns)
    for house in y.cat.categories:
        yh = (y == house)
        gradient = [partial_deriv(w[house], X, yh, j) for j in range(N)]
        gradient = np.array(gradient)
        w[house] = w[house] - alpha * gradient
    return w

def one_vs_all_epochs(X, y, epochs):
    init = pd.DataFrame(index=X.columns)
    for house in y.cat.categories:
        init[house] = pd.Series(0, index=X.columns)
        init[house]['Bias'] = 1
    w = [init] * epochs
    for i in range(1, epochs):
        w[i] = iterate_one_vs_all(X, y, w[i - 1])
    # w = pd.DataFrame(index=X.columns)
    return w

def one_vs_all(X, y, epochs=30):
    return (one_vs_all_epochs(X, y, epochs))[epochs - 1]


def main():
    try:
        parameters = pd.read_csv(sys.argv[1])
    except IndexError:
        print('Usage: python3 logreg_train.py [file]')
        sys.exit(1)
    except OSError:
        print('Missing file')
        sys.exit(1)
    X = prepare_X(parameters)
    y = parameters['Hogwarts House'].astype('category')
    w = one_vs_all(X, y)
    w.to_csv('toto')
    return

if (__name__ == "__main__"):
	main()
