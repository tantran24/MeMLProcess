import sys
import os

sys.path.append(".")

from models.LinearRegression import linear_regression
from utils.data_loader import load_data
from utils.evaluate import MSE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score as R2


def train(data_path):
    X_train, X_test, y_train, y_test = load_data(data_path)

    model = linear_regression()
    
    model.fit(X_train, y_train)
    y_predict = model.predict(X_train)
    
    plt.scatter(np.arange(0, y_train.shape[0], 1), y_train,  label = 'y^')
    plt.scatter(np.arange(0, y_predict.shape[0], 1), y_predict, label = 'y_predict', color = 'red')
    plt.title('y_predict vs y^')
    plt.xlabel('samples')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
    print('MSE on trainset : {}'.format(MSE(y_train, y_predict)))
    print('R2 on trainset : {}'.format(R2(y_train, y_predict)))

    y_predict = model.predict(X_test)
    print('MSE on testset : {}'.format(MSE(y_test, y_predict)))
    print('R2 on testset : {}'.format(R2(y_test, y_predict)))

    return model

if __name__ == "__main__":
    data_path = './data/fish.csv'
    train(data_path)