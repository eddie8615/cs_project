import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import arch
from scipy.stats import norm, skew, kurtosis
import statsmodels.stats.diagnostic
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime as dt
import pandas_datareader.data as web
from statsmodels.stats.diagnostic import acorr_ljungbox
import compareMeanModel as mm
import compareVolatility as cv

def plot(returns):
    plt.plot(returns, color='tomato', label='Daily Returns')
    plt.legend(loc='upper right')
    plt.show()

def main():
    # Retrieve data from fred
    start = dt.datetime(2010, 1, 2)
    end = dt.datetime(2020, 12, 31)
    currency_rate = web.DataReader('sp500', 'fred', start, end)
    returns = 100 * currency_rate['sp500'].pct_change().dropna()
    currency_rate['logprice'] = np.log(currency_rate.sp500).dropna()

    logreturn = currency_rate['logprice'] - currency_rate['logprice'].shift(1)

    dtf = pd.DataFrame(returns)

    # print(pd.DataFrame(returns).describe())
    print(dtf.kurtosis())
    print(kurtosis(dtf))

    plot_pacf(logreturn**2)
    plt.show()

#   From the mean model comparison, the correlation coefficients of each mean model are close to 1 which means that the impact of three models has no difference
#   So I am going to use Constant model to compare different volatility comparison

    vol_model = cv.VolatilityModel(currency_rate, returns)
    # vol_model.ljungboxtest()
    vol_model.plot()
    # vol_model.forecast()

if __name__ == "__main__":
    main()