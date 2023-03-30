import pandas as pd
from lhmind.train import train_model

# Load data from CSV files
blood_test_data = pd.read_csv("test.csv", low_memory=False)
reference_values = pd.read_csv("reference.csv")

# Define training parameters
params = {
    'hospital_phone': '1234567890',
    'train_type': 'normal_cancer',
    'blood_test_data': blood_test_data,
    'model_file_storage_path': '/Users/hermann/PycharmProjects/lhmind/models',
    'model_name': 'logistic_regression',
    'reference_values': reference_values
}

# Train the model and get evaluation results
evaluation_result = train_model(params)
