import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

class MeanModel:
    def __init__(self, data, returns):
        self.data = data
        self.returns = returns
        self.construct()

    def construct(self):
        p, q=1, 1
        vol='GARCH'
        dist= 'normal'
        armean = arch_model(self.returns, p=p, q=q, vol=vol, dist=dist, mean='AR')
        cmean = arch_model(self.returns, p=p, q=q, vol=vol, dist=dist, mean='Constant')
        zeromean = arch_model(self.returns, p=p, q=q, vol=vol, dist=dist, mean='Zero')
        self.armean_res = armean.fit()
        self.cmean_res = cmean.fit()
        self.zeromean_res = zeromean.fit()
        self.arvol = self.armean_res.conditional_volatility
        self.cvol = self.cmean_res.conditional_volatility
        self.zerovol = self.zeromean_res.conditional_volatility

    def plot(self):
        plt.plot(self.arvol, color='blue', label='AR Mean')
        plt.plot(self.cvol, color='red', label='Constant Mean')
        plt.plot(self.zerovol, color='yellow', label='Zero Mean')
        plt.legend('upper right')
        plt.show()

        print('AR Mean and Constant Mean', np.corrcoef(self.arvol, self.cvol)[0, 1])
        print('AR Mean and Zero Mean', np.corrcoef(self.arvol, self.zerovol)[0, 1])
        print('Constant Mean and Zero Mean', np.corrcoef(self.cvol, self.zerovol)[0, 1])
