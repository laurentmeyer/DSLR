import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stats as stats
import utils as utils

pd.options.display.float_format = '{:,.2f}'.format

dataset = pd.read_csv('data/dataset_train.csv')
dataset = dataset.dropna()
data = dataset.select_dtypes(include=['float64'])
features = list(dataset.select_dtypes(include=['float64']))
colors = dataset['Hogwarts House'].astype('category').cat.codes

sm = pd.plotting.scatter_matrix(data, c=colors, figsize=(18, 15))
for subaxis in sm:
        for ax in subaxis:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
plt.show()