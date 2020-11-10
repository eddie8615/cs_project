import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox

class VolatilityModel:

    def __init__(self, data, returns):
        self.data = data
        self.returns = returns
        self.construct()

    def construct(self):
        p,q = 1,1
        self.gm = arch_model(self.returns, p=3, q=3, mean='AR',vol='garch', dist='normal')
        self.egm = arch_model(self.returns, p=1, q=1, o=1, mean='AR',vol='egarch', dist='normal')
        self.gjr_gm = arch_model(self.returns, p=1, q=3, o=1, mean='AR',vol='garch', dist='normal')
        self.gm_res = self.gm.fit()
        print(self.gm_res.arch_lm_test())
        self.egm_res = self.egm.fit()
        print(self.egm_res.arch_lm_test())
        self.gjr_gm_res = self.gjr_gm.fit()
        print(self.gjr_gm_res.arch_lm_test())

        print(self.gm_res.summary())
        print(self.egm_res.summary())
        print(self.gjr_gm_res.summary())

    def ljungboxtest(self):

        print('P-values:', acorr_ljungbox(self.gm_res.std_resid, lags=10)[1])
        print('P-values:', acorr_ljungbox(self.egm_res.std_resid, lags=10)[1])
        print('P-values:', acorr_ljungbox(self.gjr_gm_res.std_resid, lags=10)[1])

    def plot(self):
        plt.plot(self.returns, color='grey', alpha=0.4, label = 'Returns')
        plt.plot(self.gm_res.conditional_volatility, color = 'gold', label = 'GARCH volatility')
        plt.plot(self.gjr_gm_res.conditional_volatility, color='red', label = 'GJR-GARCH volatility')
        plt.plot(self.egm_res.conditional_volatility, color='blue', label = 'EGARCH volatility')

        plt.legend(loc = 'upper right')
        plt.show()

    def forecast(self):
        rolling_predictions = {}
        test_size = 300
        end_loc = len(self.returns)-test_size
        train = 0
        for i in range(test_size):
            train = self.returns[:-(test_size-i)]
            print(len(train))
            model = arch_model(train, p=3, q=3)
            model_fit = model.fit(first_obs=i, last_obs=i+end_loc,disp='off')
            pred = model_fit.forecast(horizon=1).variance
            fcast = pred.iloc[i+end_loc - 1]
            rolling_predictions[fcast.name] = fcast

        rolling_predictions = pd.DataFrame(rolling_predictions).T
        plt.plot(self.returns[-test_size:], color= 'grey')
        plt.plot(rolling_predictions, color='red')
        plt.title('Volatility Prediction - Rolling forecast', fontsize = 20)
        plt.legend(['True Returns', 'Predicted'], fontsize=16)
        plt.show()
        self.evaluation(self.returns[-test_size:], rolling_predictions)

    def evaluation(self, observation, prediction):

        mse = mean_squared_error(observation, prediction)
        print("Mean squared error: {:.3g}".format(mse))

        mae = mean_absolute_error(observation, prediction)
        print("Mean absolute error: {:.3g}".format(mae))


