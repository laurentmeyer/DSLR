import numpy as np
import pandas as pd
import stats as stats
from functools import partial
from math import floor

def lerp(a, b, value):
    if a == b or value == np.nan:
        return np.nan
    return (value - a) / (b - a)

def normalize(series):
    series = series.dropna()
    s_min = stats.min(series)
    s_max = stats.max(series)
    h = partial(lerp, s_min, s_max)
    return series.map(h)

def fill_missing_data(series):
    mean = stats.mean(series)
    return series.fillna(value=mean)

def add_bias(X):
    X.loc[:,'Bias'] = pd.Series(data=np.ones(len(X.index)), index=X.index)
    return X

def cut(series, bins=10):
    series = normalize(series)
    h = lambda x : bins - 1 if x == 1. else int(floor(x * bins))
    return series.map(h)