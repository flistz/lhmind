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

    model.train(X_train, y_train)
    evaluation_result = model.evaluate(X_test, y_test)
    print(evaluation_result['report'])
    # Save the trained model
    model_filename = f"{hospital_phone}_{train_type}_{model_name}_model.pkl"
    model_file_path = os.path.join(model_file_storage_path, model_filename)
    model.save_model(model_file_path)

    # Save training information
    save_training_information(hospital_phone, train_type, model_name, evaluation_result, model_file_path)

    return evaluation_result
