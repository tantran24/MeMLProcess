from sklearn.metrics import mean_squared_error

def MSE(y, y_predict):
    return mean_squared_error(y, y_predict)