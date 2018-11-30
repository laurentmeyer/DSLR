import numpy as np
import pandas as pd
import math as math
import utils as utils
import sys as sys
import logreg_train as lt
import logreg_predict as lp

try:
    parameters = pd.read_csv(sys.argv[1], index_col=0)
except IndexError:
    print('Usage: python3 logreg_train.py [file]')
    sys.exit(1)
except OSError:
    print('Missing file')
    sys.exit(1)

def split_parameters(p):
    np.random.seed(0)
    hold_percentage = 0.25
    N = len(p.index)
    validation = np.array([False for i in range(N)])
    hold_count = (int)(hold_percentage * N)
    validation[:hold_count] = True
    np.random.shuffle(validation)
    print(validation)
    training = ~validation
    return p[training], p[validation]

training, validation = split_parameters(parameters)
Xt = lt.prepare_X(training)
Xv = lt.prepare_X(validation)
yt = training['Hogwarts House'].astype('category')
yv = validation['Hogwarts House'].astype('category')

w = lt.one_vs_all_epochs(Xt, yt, 3)
prediction = lp.predict(Xv, w)


# X = parameters.select_dtypes(include=['float64'])
# X = X.apply(utils.fill_missing_data)
# X = X.apply(utils.normalize)
# X = utils.prepend_ones(X)




# Y = parameters['Hogwarts House'].astype('category')
# houses = Y.cat.categories
# y = pd.DataFrame(columns=houses)
# for c in y.columns:
#     y[c] = Y == c

# w = np.zeros((len(houses), N))

# def g(z):
#     return 1. / (1. + np.exp(-z))

# def cost(w, X, y):
#     h = g(X.dot(w.T))
#     return sum((-np.log(h) * y - np.log(1. - h) * ~y)) / len(y)

# def partial_deriv(w, X, y, j):
#     h = g(X.dot(w.T))
#     return sum(X[X.columns[j]] * (h - y)) / len(y)

# def train(X, y, alpha=0.3, iterations=300):
#     # print('=========================')
#     w = np.zeros(len(X.columns))
#     w[0] = 1
#     i = 0
#     while (i < iterations):
#         gradient = np.array([partial_deriv(w, X, y, j) for j in range(len(w))])
#         w = w - alpha * gradient
#         # print(cost(w, X, y))
#         i = i + 1
#     return w



# # w = w[0]
# # w[0] = 1
# # y = y[houses[0]]
# # a = cost(w, X, y)
# w = np.array([train(X, y[house]) for house in houses])
# w = pd.DataFrame(data=w, index=houses, columns=X.columns)
# w = w.T
# w.to_csv('toto')
# a = g(X.dot(w))
# m = a.idxmax(axis=1)

# # print(a)