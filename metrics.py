from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error


def testScore(pred, target):
    testScore = mean_squared_error(y_pred=pred, y_true=target)
    print("MSE Score: {score}".format(score=testScore))
    root_testScore = mean_squared_error(y_pred=pred, y_true=target, squared=False)
    print("RMSE Score: {score}".format(score=root_testScore))
    mape = mean_absolute_percentage_error(y_pred=pred, y_true=target)
    print("MAPE Score: {score}".format(score=mape))
    mae = mean_absolute_error(y_pred=pred, y_true=target)
    print("MAE Score: {score}".format(score=mae))
