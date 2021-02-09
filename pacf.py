from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web
import numpy as np
import data_reader as dr
import easyargs

@easyargs
def main(infile):
    data = dr.read(infile)

    plot_pacf(data['Daily_log_return']**2)

    outfile = dr.write(infile)
    outfile = outfile.split(".")[0]+"-pacf.png"
    with open(outfile, 'w') as f:
        plt.savefig(f"{outfile}")
        plt.show()

    return None

if __name__ == "__main__":
    main()