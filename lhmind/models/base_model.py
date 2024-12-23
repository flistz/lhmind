import os
import json
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import classification_report


class BaseModel:
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]  # Output probabilities for class 1

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        print(self.__class__.__name__)
        print(classification_report(y, y_pred))
        return classification_report(y, y_pred, output_dict=True)

    def save_training_information(self, model_file_path, train_type, training_results):
        info_file_path = os.path.join(model_file_path, 'training_info.json')

        if os.path.exists(info_file_path):
            with open(info_file_path, 'r') as info_file:
                info_data = json.load(info_file)
        else:
            info_data = []

        info_data.append({
            'train_type': train_type,
            'results': training_results
        })

        with open(info_file_path, 'w') as info_file:
            json.dump(info_data, info_file)

    def save_model(self, model_file_path, evaluation_result, feature_names, median_values_dict):
        # Save the model to a file
        joblib.dump(self.model, model_file_path)

        # Prepare training information
        training_info = {
            'model_file_path': model_file_path,
            'evaluation_result': evaluation_result,
            'feature_names': feature_names,
            'median_values_dict': median_values_dict
        }

        # Load existing training information from the JSON file
        training_info_file = os.path.join(os.path.dirname(model_file_path), 'training_info.json')
        if os.path.exists(training_info_file):
            with open(training_info_file, 'r') as f:
                existing_training_info = json.load(f)
        else:
            existing_training_info = {}

        # Update the existing training information with the new training_info
        existing_training_info[self.__class__.__name__] = training_info

        # Save the updated training information to the JSON file
        with open(training_info_file, 'w') as f:
            json.dump(existing_training_info, f, indent=4)

    def load_model(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def load_training_info(self, model_file_path):
        # Construct the training_info.json file path
        training_info_file_path = os.path.join(os.path.dirname(model_file_path), 'training_info.json')

        # Load the training information
        with open(training_info_file_path, 'r') as f:
            training_info = json.load(f)

        return training_info[self.__class__.__name__]
