import numpy as np
from sklearn.linear_model import LinearRegression
from models.BaseModel import BaseModel


class linear_regression(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)