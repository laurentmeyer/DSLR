#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math as math
import utils as utils
import sys as sys

def g(z):
    return 1. / (1. + np.exp(-z))

def cost(X, y, w):
    cost = 0.
    h = g(X.dot(w))
    for house in y.cat.categories:
        yh = (y == house)
        Xh = h[house]
        cost = cost + sum((-np.log(Xh) * yh - np.log(1. - Xh) * ~yh)) / len(yh)
    return cost

def partial_deriv(w, X, y, j):
    h = g(X.dot(w.T))
    return sum(X[X.columns[j]] * (h - y)) / len(y)

def iterate_one_vs_all(X, y, w, alpha=1.):
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
    return w

def one_vs_all(X, y, epochs=200):
    e = one_vs_all_epochs(X, y, epochs)
    return e[epochs - 1]

def main():
    try:
        parameters = pd.read_csv(sys.argv[1])
    except IndexError:
        print('Usage: python3 logreg_train.py [file]')
        sys.exit(1)
    except OSError:
        print('Missing file')
        sys.exit(1)
    X = utils.prepare_X(parameters)
    y = parameters['Hogwarts House'].astype('category')
    w = one_vs_all(X, y)
    w.to_csv('weights.csv')
    return

if (__name__ == "__main__"):
	main()
