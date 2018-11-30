import pandas as pd
import numpy as np
from sys import argv
import stats as stats

def describe(dataset):
    dataset = dataset.dropna(axis='columns', how='all')
    dataset = dataset.select_dtypes(include=['float64'])
    desc = pd.DataFrame({
        'Count': dataset.apply(stats.count),
        'Mean': dataset.apply(stats.mean),
        'Std': dataset.apply(stats.std),
        'Min': dataset.apply(stats.min),
        '25%': dataset.apply(stats.quartile_1),
        '50%': dataset.apply(stats.quartile_2),
        '75%': dataset.apply(stats.quartile_3),
        'Max': dataset.apply(stats.max),
    })
    print(desc.T)

def main():
    pd.options.display.float_format = '{:,.2f}'.format
    if len(argv) != 2:
        print("usage: python3 decribe.py [filepath]")
        return
    try:
        dataset = pd.read_csv(argv[1])
        describe(dataset)
    except FileNotFoundError:
        print("Invalid argument: file not found")
    except pd.errors.ParserError:
        print("Invalid argument: file not readable as CSV")

if (__name__ == "__main__"):
	main()