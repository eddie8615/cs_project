from arch import arch_model
import matplotlib.pyplot as plt
import pandas as pd
from arch import arch_model
import easyargs
import data_reader as dr

@easyargs
def main(infile):
    data, source = dr.read(infile)
    if source == 'kospi':
        model = arch_model(data['Daily_log_return'], p=1, q=1, vol='garch')
    else:
        model = arch_model(data['Daily_log_return'], p=2, q=2, vol='garch')

    model_result = model.fit()
    print(model_result.summary())
    print(model_result.arch_lm_test())
    plt.plot(data['Daily_log_return'], alpha=0.4, color='grey', label='return')
    plt.plot(model_result.conditional_volatility, color='red', label='GARCH')
    plt.show()


if __name__ == "__main__":
    main()