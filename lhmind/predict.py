import json
import os
from .data_processing import preprocess_data_for_prediction
import pickle
import json
from .models import LogisticRegressionModel, RandomForestModel, SVMModel


def predict(params):
    hospital_phone = params['hospital_phone']
    train_type = params['train_type']
    disease_type = params['disease_type']
    blood_test_data = params['blood_test_data']
    model_file_storage_path = params['model_file_storage_path']
    model_name = params['model_name']
    reference_values = params['reference_values']

    # Instantiate the appropriate model
    if model_name == 'logistic_regression':
        model = LogisticRegressionModel()
    elif model_name == 'random_forest':
        model = RandomForestModel()
    elif model_name == 'svm':
        model = SVMModel()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    # Create a directory structure for the model file
    hospital_directory = os.path.join(model_file_storage_path, hospital_phone)
    train_type_directory = os.path.join(hospital_directory, train_type)
    disease_type_directory = os.path.join(train_type_directory, disease_type)

    # Load the trained model
    model_filename = f"{model_name}_model.pkl"
    model_file_path = os.path.join(disease_type_directory, model_filename)
    model.load_model(model_file_path)

    # Load the training information
    training_info = model.load_training_info(model_file_path)

    # Preprocess the data for prediction
    blood_test_data_preprocessed = preprocess_data_for_prediction(blood_test_data, reference_values, training_info['median_values_dict'], training_info['feature_names'])

    # Make predictions
    prediction_results = model.score(blood_test_data_preprocessed)

    return prediction_results

