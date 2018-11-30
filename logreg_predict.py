import numpy as np
import pandas as pd
import math as math
import utils as utils
import sys as sys
import logreg_train as lt

def predict(X, w):
    X = X.dot(w)
    X = 1. / (1. + np.exp(-X))
    X = X.idxmax(axis=1)
    X = X.rename('Hogwarts House')
    return X

def main():
    try:
        X = pd.read_csv(sys.argv[1])
        w = pd.read_csv(sys.argv[2], index_col=0)
    except IndexError:
        print('Usage: python3 logreg_predict.py [to_categorize] [weights]')
        sys.exit(1)
    except OSError:
        print('Missing file')
        sys.exit(1)
    X = lt.prepare_X(X)
    X = predict(X, w)
    X.to_csv(path='houses.csv', header=True, index_label='Index')
    return


# if (__name__ == "__main__"):
	# main()