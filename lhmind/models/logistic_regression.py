# lhmind/models/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(LogisticRegression(max_iter=1000))

