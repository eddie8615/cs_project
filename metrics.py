from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, r2_score


def testScore(pred, target):
    testScore = mean_squared_error(pred, target)
    print("test Score: {score} MSE".format(score=testScore))
    root_testScore = mean_squared_error(pred, target, squared=False)
    print("test Score: {score} RMSE".format(score=root_testScore))
    mape = mean_absolute_percentage_error(pred, target)
    print("test Score: {score} MAPE".format(score=mape))
    r2_test = r2_score(target, pred)
    print("test Score: {score} R2 score".format(score=r2_test))
