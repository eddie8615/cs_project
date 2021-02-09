import pandas as pd
import os
import numpy as np


def read(infile):
    root = os.getcwd()
    infile = os.path.join(root, 'dataset', infile)
    file_name = infile.split('/')[-1]
    print(f"Loading data from file {infile}...\n")
    with open(infile, 'r') as f:
        data = pd.read_csv(infile)

    data['Daily_log_return'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
    data['Daily_return'] = 100 * data['Close'].pct_change()
    data = data.dropna()
    return data, file_name.split('.')[0]


def write(infile):
    root = os.getcwd()
    # file_name = infile.split('/')[-1]
    outfile = os.path.join(root, 'images', infile)
    print(f"Write path is {outfile}\n")
    return outfile
