import pandas as pd
import os
import numpy as np
import math


def read():
    # root = os.getcwd()
    # infile = os.path.join(root, 'dataset', infile)
    # file_name = infile.split('/')[-1]
    # print(f"Loading data from file {infile}...\n")
    # with open(infile, 'r') as f:
    #     data = pd.read_csv(infile)

    price = pd.read_csv('dataset/kospi.csv')
    oil = pd.read_csv('dataset/oilprice.csv')
    gold = pd.read_csv('dataset/goldprice.csv')
    data = pd.DataFrame()
    data['Date'] = price['Date']
    data['Daily_trading_range'] = price['High'] - price['Low']
    data['Log_Volume_change'] = np.log((price['Volume'] / price['Volume'].shift(1)))*100
    data['Daily_return'] = price['Close'].pct_change().dropna()
    data['Daily_log_return'] = np.log(price['Close'] / price['Close'].shift(1))
    data['Index'] = price['Close']
    data['Past_vol22'] = data['Daily_log_return'].rolling(window=22).std() * np.sqrt(252)

    data['gold'] = gold['Close']
    data['oil'] = oil['Close']

    data = data.dropna()

    volatility = np.sqrt((data['Daily_log_return']**2).rolling(window=22).sum()/21)*np.sqrt(252)
    vol10 = np.sqrt((data['Daily_log_return']**2).rolling(window=10).sum()/9)*np.sqrt(252)
    # target = yz_vol_measure(data)
    # target10 = yz_vol_measure(data, window=10)
    target22 = pd.DataFrame(volatility)
    target10 = pd.DataFrame(vol10)

    data['Target22'] = target22
    data['Target10'] = target10
    # data['Target10'] = target10

    return data

# This method refers to Yang-Zhang volatility measure for making ground truth of the volatility
def yz_vol_measure(data, window=22, trading_periods=252):
    nHigh = np.log(data['High'] / data['Open'])
    high2 = np.log(data['High']/data['Close'])
    nLow = np.log(data['Low'] / data['Open'])
    low2 = np.log(data['Low']/ data['Close'])
    nClose = np.log(data['Close'] / data['Open'])

    log_oc = np.log(data['Open'] / data['Close'].shift(1))
    log_oc_sq = log_oc ** 2

    log_cc = np.log(data['Close'] / data['Close'].shift(1))
    log_cc_sq = log_cc ** 2

    # rs = nHigh * (nHigh - nClose) + nLow * (nLow - nClose)
    rs = (high2 * (nHigh - nClose) + low2 * (nLow - nClose))/window

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

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    return result


def write(infile):
    root = os.getcwd()
    # file_name = infile.split('/')[-1]
    outfile = os.path.join(root, 'images', infile)
    print(f"Write path is {outfile}\n")
    return outfile
