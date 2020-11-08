import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

class VolatilityModel:

    def __init__(self, data, returns):
        self.data = data
        self.returns = returns
        self.construct()

    def construct(self):
        p,q = 1,1
        gm = arch_model(self.returns, p=p, q=q, vol='garch', dist='skewt')
        egm = arch_model(self.returns, p=p, q=q, o=1, vol='egarch', dist='skewt')
        gjr_gm = arch_model(self.returns, p=p, q=q, o=1,vol='garch', dist='skewt')
        self.gm_res = gm.fit()
        self.egm_res = egm.fit()
        self.gjr_gm_res = gjr_gm.fit()


    def plot(self):
        plt.plot(self.returns, color='grey', alpha=0.4, label = 'Returns')
        plt.plot(self.gm_res.conditional_volatility, color = 'gold', label = 'GARCH volatility')
        plt.plot(self.gjr_gm_res.conditional_volatility, color='red', label = 'GJR-GARCH volatility')
        plt.plot(self.egm_res.conditional_volatility, color='blue', label = 'EGARCH volatility')

        plt.legend(loc = 'upper right')
        plt.show()

    def forecast(self):
        train, test = self.returns[:len(self.returns)*0.9], self.returns[len(self.returns)*0.1 + 1]

        

