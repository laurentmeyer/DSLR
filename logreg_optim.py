#!/usr/bin/env python3
import numpy as np
import pandas as pd
import math as math
import utils as utils
import sys as sys
import logreg_train as lt
import logreg_predict as lp
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

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
    training = ~validation
    t = p[training]
    v = p[validation]
    Xt = utils.prepare_X(t)
    Xv = utils.prepare_X(v)
    yt = t['Hogwarts House'].astype('category')
    yv = v['Hogwarts House'].astype('category')
    return Xt, yt, Xv, yv

Xt, yt, Xv, yv = split_parameters(parameters)

N = 1000
w = lt.one_vs_all_epochs(Xt, yt, N)
cost_training = np.array([lt.cost(Xt, yt, i) for i in w])
cost_validation = np.array([lt.cost(Xv, yv, i) for i in w])
plt.plot(np.arange(1., N + 1., 1.), cost_training, cost_validation, 'r')
plt.show()

N = 200
w = lt.one_vs_all_epochs(Xt, yt, N)
prediction = lp.predict(Xv, w[N - 1])
same = (prediction == yv)
print("precision = {}".format(sum(same) / len(same)))