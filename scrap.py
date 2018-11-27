import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stats as stats
import utils as utils

pd.options.display.float_format = '{:,.2f}'.format
dataset = pd.read_csv('data/dataset_train.csv')

bin_count = 20

houses = list(dataset['Hogwarts House'].dropna().unique())
subjects = list(dataset.select_dtypes(include=['float64']))
subject = subjects[1]

fig = plt.figure(figsize=(12, 10))
fig.suptitle('Comparison for each subject', fontsize=16)
gs = fig.add_gridspec(4, 4)

for i, s in enumerate(subjects):
    ax = fig.add_subplot(gs[i % 4, i // 4])
    ax.set_xlabel(s)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False)
    subject_min = stats.min(dataset[s])
    subject_max = stats.max(dataset[s])
    bins = np.linspace(subject_min, subject_max, bin_count)
    for house in houses:
        mask = (dataset['Hogwarts House'] == house)
        data = dataset[s]
        data = data[mask]
        plt.hist(data, bins, alpha=0.2)
plt.legend(houses, bbox_to_anchor=(1.1, -0.5))
plt.show()