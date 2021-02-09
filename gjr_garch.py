import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
import easyargs
import data_reader as dr

@easyargs
def main(infile):
    data, source = dr.read(infile)
    data = pd.DataFrame(data)
    if source == 'kospi':
        model = arch_model(data['Daily_log_return'], p=1, q=2, o=1, vol='garch')
    else:
        model = arch_model(data['Daily_log_return'], p=1, q=1, o=1, vol='garch')

    model_result = model.fit()
    print(model_result.summary())
    plt.plot(data['Daily_log_return'], color='grey', alpha=0.4, label='Returns')
    plt.plot(model_result.conditional_volatility, color='red', label='gjr-garch')
    plt.show()


def plot(data, model_result):
    plt.plot(data['Daily_log_return'], color='grey', alpha=0.4, label = 'Returns')
    plt.plot(model_result.conditional_volatility, color='red', label='gjr-garch')
    plt.show()


if __name__ == "__main__":
    main()