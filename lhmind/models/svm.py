# lhmind/models/svm.py
from sklearn.svm import SVC
from .base_model import BaseModel

class SVMModel(BaseModel):
    def __init__(self):
        self.model = SVC()
