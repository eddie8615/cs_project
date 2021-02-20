import pandas as pd
import os
import numpy as np
import math


def read(infile):
    root = os.getcwd()
    infile = os.path.join(root, 'dataset', infile)
    file_name = infile.split('/')[-1]
    print(f"Loading data from file {infile}...\n")
    with open(infile, 'r') as f:
        data = pd.read_csv(infile)

    data = data.dropna()

    data['Daily_log_return'] = 100 * np.log(data['Close'] / data['Close'].shift(1))
    data['Daily_trading_range'] = data['High'] - data['Low']
    data['Log_Volume_change'] = 100 * np.log(data['Volume'] / data['Volume'].shift(1))
    data['Daily_return'] = 100 * data['Close'].pct_change()


    target = yz_vol_measure(data)
    data['Target'] = target

    return data, file_name.split('.')[0]


# This method refers to Yang-Zhang volatility measure for making ground truth of the volatility
def yz_vol_measure(data, window=22, trading_periods=252):
    nHigh = np.log(data['High'] / data['Open'])
    nLow = np.log(data['Low'] / data['Open'])
    nClose = np.log(data['Close'] / data['Open'])

    log_oc = np.log(data['Open'] / data['Close'].shift(1))
    log_oc_sq = log_oc ** 2

    log_cc = np.log(data['Close'] / data['Close'].shift(1))
    log_cc_sq = log_cc ** 2

    rs = nHigh * (nHigh - nClose) + nLow * (nLow - nClose)

    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    return result


def write(infile):
    root = os.getcwd()
    # file_name = infile.split('/')[-1]
    outfile = os.path.join(root, 'images', infile)
    print(f"Write path is {outfile}\n")
    return outfile
