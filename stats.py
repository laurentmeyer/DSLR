import numpy as np
import pandas as pd
from functools import reduce
from functools import partial
from math import sqrt

def count(series):
    return len(series.dropna())

def sum(series):
    series = series.dropna()
    return reduce(lambda x, y: x + y, series)

def mean(series):
    return sum(series) / count(series)

def std(series):
    series = series.dropna()
    series = (series - mean(series)) ** 2
    return sqrt(sum(series))

def min(series):
    series = series.dropna()
    return reduce(lambda x, y: x if x < y else y, series)

def max(series):
    series = series.dropna()
    return reduce(lambda x, y: x if x > y else y, series)

def percent(p, series):
    series = series.dropna()
    series = series.sort_values()
    position = int(len(series) * p / 4)
    return series.iat[position]

quartile_1 = partial(percent, 0.25)
quartile_2 = partial(percent, 0.50)
quartile_3 = partial(percent, 0.75)