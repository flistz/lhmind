# lhmind/train.py
import os
from lhmind.data_processing import preprocess_data
from lhmind.models import LogisticRegressionModel, RandomForestModel, SVMModel
from lhmind.utils import save_training_information


def train_model(params):
    hospital_phone = params['hospital_phone']
    train_type = params['train_type']
    blood_test_data = params['blood_test_data']
    model_file_storage_path = params['model_file_storage_path']
    model_name = params['model_name']
    reference_values = params['reference_values']

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(blood_test_data, train_type, reference_values)

    # Train the model and get evaluation results
    if model_name == 'logistic_regression':
        model = LogisticRegressionModel()
    elif model_name == 'random_forest':
        model = RandomForestModel()
    elif model_name == 'svm':
        model = SVMModel()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")

    model.fit(X_train, y_train)
    evaluation_result = model.evaluate(X_test, y_test)

    # Create a directory structure for the model file
    hospital_directory = os.path.join(model_file_storage_path, hospital_phone)
    train_type_directory = os.path.join(hospital_directory, train_type)
    os.makedirs(train_type_directory, exist_ok=True)

    # Save the trained model
    model_filename = f"{model_name}_model.pkl"
    model_file_path = os.path.join(train_type_directory, model_filename)

    # Extract the feature names
    feature_names = X_train.columns.tolist()

    # Save the model and training information
    model.save_model(model_file_path, evaluation_result, feature_names)

    return evaluation_result
