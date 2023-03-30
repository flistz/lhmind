# lhmind/utils.py
import csv
import os

def save_training_information(hospital_phone, train_type, model_name, evaluation_result, model_file_path):
    training_info_file = "training_information.csv"

    if not os.path.exists(training_info_file):
        with open(training_info_file, "w", newline='') as csvfile:
            fieldnames = ['hospital_phone', 'train_type', 'evaluation_result', 'model_file_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(training_info_file, "a", newline='') as csvfile:
        fieldnames = ['hospital_phone', 'train_type', 'evaluation_result', 'model_file_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'hospital_phone': hospital_phone,
            'train_type': train_type,
            'evaluation_result': evaluation_result,
            'model_file_path': model_file_path
        })
