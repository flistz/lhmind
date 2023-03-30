# lhmind/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()
