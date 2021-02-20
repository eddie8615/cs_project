import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import data_reader as dr
import easyargs

@easyargs
def main(infile):
    data, source = dr.read(infile)
    data = data.drop(['Adj Close'], axis=1)
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    print(len(train))
    print(len(test))


    # df_for_training = data[cols].astype(float)
    #
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    trainX = []
    trainY = []


    # n_futures = 10
    # n_past = 22
    #
    # for i in range(n_past, len(df_for_training_scaled) - n_futures + 1):
    #     trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    #
    # trainX = np.array(trainX)
    #
    # model = Sequential()
    # model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    # model.add(LSTM(32, activation='relu', return_sequences=False))
    # model.add(Dropout(0.2))
    # # model.add(Dense())
    #
    # model.compile(optimizer='adam', loss='mse')


if __name__ == '__main__':
    main()