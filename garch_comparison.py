import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from scipy.stats import norm
import datetime as dt
import pandas_datareader.data as web
import compareMeanModel as mm
import compareVolatility as cv

def plot(returns):
    plt.plot(returns, color='tomato', label='Daily Returns')
    plt.legend(loc='upper right')
    plt.show()

def main():
    # Retrieve data from fred
    start = dt.datetime(2010, 10, 30)
    end = dt.datetime(2020, 10, 30)
    currency_rate = web.DataReader('DEXKOUS', 'fred', start, end)
    des = currency_rate['DEXKOUS'].describe()
    logReturns = currency_rate['DEXKOUS'].apply(np.log).dropna()
    returns = 100 * currency_rate['DEXKOUS'].pct_change().dropna()


    # meanModel = mm.MeanModel(currency_rate, returns)
    # meanModel.plot()
#   From the mean model comparison, the correlation coefficients of each mean model are close to 1 which means that the impact of three models has no difference
#   So I am going to use Constant model to compare different volatility comparison

    vol_model = cv.VolatilityModel(currency_rate, returns)
    vol_model.plot()

if __name__ == "__main__":
    main()