import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.optimizers import Adam
import pandas as pd
from arch import arch_model
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, r2_score
from tensorflow import keras
import seaborn as sns
import easyargs


def main():
    data = read_data()
    data_cleaned = data[22:].dropna().reset_index(drop=True)
    split_index = int(len(data_cleaned)*0.8)

    print('split: ', split_index)
    # create_ewma(data_cleaned, split_index)
    # ewma_estim = ewma_estimation(data_cleaned)
    # garch_estim = create_garch(data, split_index)
    gjr_estim = create_gjr(data, split_index)
    # data_cleaned['EWMA'] = ewma_estim['EWMA']
    # data_cleaned['GARCH'] = garch_estim['GARCH']
    data_cleaned['GJR'] = gjr_estim['GJR']

    Xtrain, ytrain, Xtest, ytest, scaler = data_prep(data_cleaned, split_index)

    run_lstm(Xtrain, ytrain, Xtest, ytest, scaler)


def run_lstm(Xtrain, ytrain, Xtest, ytest, scaler):
    lstm = create_lstm(Xtrain)
    lstm_fit = lstm.fit(Xtrain, ytrain, batch_size=16, epochs=150)
    lstm_forecast = lstm.predict(Xtest)

    rev_forecast = scaler.inverse_transform(lstm_forecast)
    rev_ytest = scaler.inverse_transform(ytest)

    plot(rev_forecast, rev_ytest)
    evaluate(rev_forecast, rev_ytest, 'LSTM Evaluation')


def run_ann(Xtrain, ytrain, Xtest, ytest, scaler):
    model = create_ann(Xtrain)
    model_fit = model.fit(Xtrain, ytrain, batch_size=16, epochs=50)
    forecast = model.predict(Xtest)

    rev_forecast = scaler.inverse_transform(forecast)
    rev_ytest = scaler.inverse_transform(ytest)

    plt.plot(rev_forecast, color='red', label='forecast')
    plt.plot(rev_ytest, color='gold', label='target')
    plt.legend()
    plt.show()
    evaluate(rev_forecast, rev_ytest, 'GER-FNN evaluation')


def data_prep(data, split_index):
    data = data.drop(['Daily_return', 'Past_vol22', 'Target10','Date'], axis=1)
    window = 22
    y_values = data[['Target22']]
    x_values = data.drop(['Target22'], axis=1)
    print(x_values.info())

    scaler = MinMaxScaler()
    scaled_x = scaler.fit_transform(x_values)
    scaled_y = scaler.fit_transform(y_values)

    trainX = np.array(scaled_x[:split_index])
    testX = np.array(scaled_x[split_index:])
    trainY = np.array(scaled_y[:split_index])
    testY = np.array(scaled_y[split_index:])

    Xtrain =[]
    ytrain =[]
    Xtest =[]
    ytest = []

    for i in range(window, len(trainX)):
        Xtrain.append(trainX[i - window:i, :trainX.shape[1]])
        ytrain.append(trainY[i])
    for i in range(window, len(testX)):
        Xtest.append(testX[i - window:i, :testX.shape[1]])
        ytest.append(testY[i])

    Xtrain, ytrain = (np.array(Xtrain), np.array(ytrain))
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

    Xtest, ytest = (np.array(Xtest), np.array(ytest))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

    print(Xtrain.shape)
    print(ytrain.shape)
    print("-----")
    print(Xtest.shape)
    print(ytest.shape)
    return Xtrain, ytrain, Xtest, ytest, scaler


def create_lstm(Xtrain):
    model = Sequential()
    model.add(LSTM(units=32, return_sequences=False, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dropout(0.5))
    # model.add(Dense(20, activation='tanh'))
    # model.add(Dense(5, activation='tanh'))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt)

    return model


def create_ann(Xtrain):
    model = Sequential()
    model.add(Flatten(input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    adam = Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def create_ewma(data, split_index):

    # Data processing
    sqreturn = np.array(data['Daily_log_return']**2)
    target = np.array(data[['Target22']])
    # target = pd.DataFrame(target)

    trainX = sqreturn[:split_index]
    trainY = target[:split_index]

    prev_vol = train_ewma(trainX, trainY)

    testX = sqreturn[split_index:]
    testY = target[split_index:]

    predict_ewma(testX, testY, prev_vol)


def train_ewma(trainX, trainY):
    pred_vol = []
    weight = 0.94
    pred_vol.append(trainX[0])
    for i in range(1, len(trainX)):
        pred = weight*(pred_vol[-1]**2) + (1-weight)*trainX[i]
        pred_vol.append(np.sqrt(pred))

    pred_vol = pd.DataFrame(pred_vol) * np.sqrt(252)
    plot(pred_vol[:-1], trainY[1:])
    evaluate(pred_vol[:-1], trainY[1:], 'EWMA training')
    return pred_vol.iloc[-1]


def predict_ewma(testX, testY, prev_vol):
    weight = 0.94
    test_pred = []
    test_pred.append(prev_vol[0]/np.sqrt(252))

    for i in range(len(testX)-1):
        test_vol = weight*(test_pred[-1]**2) + (1-weight)*testX[i]
        test_pred.append(np.sqrt(test_vol))

    test_pred = pd.DataFrame(test_pred)
    test_pred = np.sqrt(252)*test_pred

    plot(test_pred, testY)
    evaluate(test_pred, testY,'EWMA prediction')


def ewma_estimation(data):
    sqreturn = np.array(data['Daily_log_return']**2)
    estimation = []
    estimation.append(data['Past_vol22'].iloc[0] / np.sqrt(252))
    weight = 0.94
    for i in range(1,len(data)):
        pred = weight*(estimation[-1]**2) + (1-weight)*(sqreturn[i-1])
        estimation.append(np.sqrt(pred))

    estimation = np.sqrt(252)*pd.DataFrame(estimation, columns=['EWMA'])

    plot(estimation, np.array(data['Target22']))
    print('-----EWMA estimation done-----')
    return estimation


def create_gjr(data, split_index):
    data = data[22:].dropna()
    data_cleaned = data[22:].dropna()
    logreturns = np.array(data[['Daily_log_return']].dropna())
    target = np.array(data_cleaned[['Target22']].dropna())

    gjr_pred = []
    for i in range(len(data)):
        train = logreturns[:i + 22] * 100
        gm = arch_model(train, p=1, q=1, o=1)
        gm_fit = gm.fit(disp='off')
        pred = gm_fit.forecast(horizon=1)
        gjr_pred.append(np.sqrt(pred.variance.values[-1, :][0]) * 0.01 * np.sqrt(252))

    print('garch pred length: ', len(gjr_pred))
    print('target length: ', len(target))
    plot(gjr_pred, target)
    gjr_pred = pd.DataFrame(gjr_pred, columns=['GJR'])
    print('-----GJR-GARCH estimation done-----')
    return gjr_pred


def create_garch(data, split_index):

    data = data[22:].dropna()
    data_cleaned = data[22:].dropna()
    logreturns = np.array(data[['Daily_log_return']].dropna())
    target = np.array(data_cleaned[['Target22']].dropna())

    garch_pred = []
    for i in range(len(data)):
        train = logreturns[:i+22]*100
        gm = arch_model(train, p=1, q=1)
        gm_fit = gm.fit(disp='off')
        pred = gm_fit.forecast(horizon=1)
        garch_pred.append(np.sqrt(pred.variance.values[-1,:][0])*0.01*np.sqrt(252))

    print('garch pred length: ', len(garch_pred))
    print('target length: ', len(target))
    plot(garch_pred, target)
    garch_pred = pd.DataFrame(garch_pred, columns=['GARCH'])
    print('-----GARCH estimation done-----')
    return garch_pred


def plot(predict, target):
    plt.plot(predict, label='predict')
    plt.plot(target, label='target')
    plt.legend()
    plt.show()


def evaluate(predict, target, title):
    print('--------'+title+'----------')
    testScore = mean_squared_error(predict, target)
    print("test Score: {score} MSE".format(score=testScore))
    root_testScore = mean_squared_error(y_pred=predict, y_true=target, squared=False)
    print("test Score: {score} RMSE".format(score=root_testScore))
    mape = mean_absolute_percentage_error(y_pred=predict, y_true=target)
    print("test Score: {score} MAPE".format(score=mape))
    r2_test = r2_score(y_true=target, y_pred=predict)
    print("test Score: {score} R2 score".format(score=r2_test))


def read_data():
    price = pd.read_csv('dataset/kospi.csv')
    oil = pd.read_csv('dataset/oilprice.csv')
    gold = pd.read_csv('dataset/goldprice.csv')
    data = pd.DataFrame()
    data['Date'] = price['Date']
    data['Daily_trading_range'] = price['High'] - price['Low']
    data['Log_Volume_change'] = np.log((price['Volume'] / price['Volume'].shift(1))) * 100
    data['Daily_return'] = price['Close'].pct_change().dropna()
    data['Daily_log_return'] = np.log(price['Close'] / price['Close'].shift(1))
    data['Past_vol22'] = np.sqrt((data['Daily_log_return']**2)).rolling(window=22).std() * np.sqrt(252)
    data['Index'] = price['Close']
    data['gold'] = gold['Close']
    data['oil'] = oil['Close']

    volatility = np.sqrt((data['Daily_log_return'] ** 2).rolling(window=22).sum() / 22) * np.sqrt(252)
    vol10 = np.sqrt((data['Daily_log_return'] ** 2).rolling(window=10).sum() / 10) * np.sqrt(252)
    # target = yz_vol_measure(data)
    # target10 = yz_vol_measure(data, window=10)
    target22 = pd.DataFrame(volatility)
    target10 = pd.DataFrame(vol10)

    data['Target22'] = target22
    data['Target10'] = target10
    # data['Target10'] = target10
    # data = data.dropna()

    return data


if __name__ == '__main__':
    main()
