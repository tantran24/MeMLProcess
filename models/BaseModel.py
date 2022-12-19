from abc import abstractclassmethod

class BaseModel:
    @abstractclassmethod
    def __init__(self):
        raise NotImplementedError

    @abstractclassmethod
    def fit(self):
        raise NotImplementedError

    @abstractclassmethod
    def predict(self):
        raise NotImplementedError

    