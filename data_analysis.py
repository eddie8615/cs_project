from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib as plt
import data_reader as dr
import easyargs
import pandas as pd


@easyargs
def main(infile):
    data, source = dr.read(infile)
    # reading data from data_reader at this file is not cleaned due to producing GARCH and EGARCH forecasts
    data = data.dropna()
    print(data)
    dtf = pd.DataFrame(data['Daily_log_return'])

    # print(dtf.describe())
    # print("----------Kurtosis-----------")
    # print(dtf.kurtosis())
    # print("----------Skewness-----------")
    # print(dtf.skew())

    print("----------ADF test-----------")
    adfTest = adfuller(data.loc[:, 'Daily_log_return'].values, autolag='AIC')
    dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic', 'P-Value', '# Lags Used', '# Observations Used'])
    for key, value in adfTest[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    print(dfResults)


if __name__ == '__main__':
    main()