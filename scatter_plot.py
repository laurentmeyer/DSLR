import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stats as stats
import utils as utils

pd.options.display.float_format = '{:,.2f}'.format

dataset = pd.read_csv('data/dataset_train.csv')
features = list(dataset.select_dtypes(include=['float64']))
feature_pairs = [(features[i], features[j])
        for i in range(len(features))
        for j in range(len(features))
        if j > i]

fig = plt.figure(figsize=(18, 15))
fig.suptitle('Comparison for each feature', fontsize=16)
gs = fig.add_gridspec(9, 9)

for i, (a, b) in enumerate(feature_pairs):
    ax = fig.add_subplot(gs[i % 9, i // 9])
    ax.set_xlabel(a + ' / ' + b, fontsize=6)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    data = dataset[[a, b]]
    data = data.dropna()
    x = data[a]
    y = data[b]
    plt.scatter(x, y)
plt.show()