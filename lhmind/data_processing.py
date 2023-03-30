# lhmind/data_processing.py
from sklearn.model_selection import train_test_split
from .constants import basic_information, blood_routine_indicators, blood_biochemical_indicators

import numpy as np


def validate_and_filter_data(data, train_type):
    filtered_data = data.copy()

    # 1. Filter rows based on the DISEASE column
    disease_filter = lambda x: x == 'normal' or (isinstance(x, str) and x[0].isupper() and x[-1].isdigit())
    filtered_data = filtered_data[filtered_data['DISEASE'].apply(disease_filter)]

    # Filter and map data based on the train_type
    filtered_data = filter_and_map_data(filtered_data, train_type)

    # 2. Remove rows with invalid feature data
    feature_names = list(basic_information.keys()) + list(blood_routine_indicators.keys()) + list(blood_biochemical_indicators.keys())
    invalid_value_filter = lambda x: np.isnan(x) or isinstance(x, (int, float))
    for feature_name in feature_names:
        filtered_data = filtered_data[filtered_data[feature_name].apply(invalid_value_filter)]

    return filtered_data


def normalize_data(data, reference_values):
    normalized_data = data.copy()

    # Normalize age
    normalized_data['AGE'] = normalized_data['AGE'].apply(lambda x: (x - 1) / (100 - 1))

    # Combine feature names from both dictionaries
    feature_names = list(blood_routine_indicators.keys()) + list(blood_biochemical_indicators.keys())

    # Initialize the dictionary to store median values for the label 0 group
    median_values_dict = {}

    # Get label 0 data
    label_0_data = normalized_data[normalized_data['label'] == 0]

    # Calculate median values for label 0 group
    for feature_name in feature_names:
        median_values_dict[feature_name] = label_0_data[feature_name].median()

    # Normalize blood test data based on gender, disease, and reference values
    for (gender, disease), group_data in normalized_data.groupby(['SEX', 'DISEASE']):
        if gender in ['M', 'F']:
            for feature_name in feature_names:
                # Get the reference values for the current feature and gender
                ref_row = reference_values.loc[(reference_values['indicator'] == feature_name) & (reference_values['gender'] == gender)]
                min_val, max_val = ref_row['min'].values[0], ref_row['max'].values[0]

                # Fill missing values in the current disease group using the median value for that group
                if group_data[feature_name].notna().any():  # Check if there's at least one non-null value
                    median_val = group_data[feature_name].median()
                else:
                    median_val = (min_val + max_val) / 2

                normalized_data.loc[(normalized_data['SEX'] == gender) & (normalized_data['DISEASE'] == disease), feature_name] = normalized_data.loc[(normalized_data['SEX'] == gender) & (normalized_data['DISEASE'] == disease), feature_name].fillna(median_val)

                # Normalize the data in-place for the current feature and gender
                mask = (normalized_data['SEX'] == gender) & (normalized_data['DISEASE'] == disease)
                normalized_data.loc[mask, feature_name] = (normalized_data.loc[mask, feature_name] - min_val) / (max_val - min_val)

    return normalized_data, median_values_dict


def filter_and_map_data(data, target_col):
    if target_col == 'normal_cancer':
        filter_func = lambda x: x == 'normal' or (isinstance(x, str) and x.startswith('C'))
        map_func = lambda x: 0 if x == 'normal' else 1
    elif target_col == 'normal_inflam':
        filter_func = lambda x: x == 'normal' or (isinstance(x, str) and not x.startswith('C') and x[0].isupper())
        map_func = lambda x: 0 if x == 'normal' else 1
    elif target_col == 'inflam_cancer':
        filter_func = lambda x: (isinstance(x, str) and x.startswith('C')) or (isinstance(x, str) and not x.startswith('C') and x[0].isupper())
        map_func = lambda x: 0 if not x.startswith('C') else 1
    elif target_col == 'normal_inflam&cancer':
        filter_func = lambda x: x == 'normal' or (isinstance(x, str) and x[0].isupper())
        map_func = lambda x: 0 if x == 'normal' else 1
    else:
        raise ValueError(f"Invalid target_col: {target_col}")

    # Filter rows based on the DISEASE column
    filtered_data = data[data['DISEASE'].apply(filter_func)]

    # Map the DISEASE column values based on the target_col
    filtered_data_with_label = filtered_data.copy()
    filtered_data_with_label['label'] = filtered_data['DISEASE'].apply(map_func)

    return filtered_data_with_label


def preprocess_data(data, train_type, reference_values):
    # Validate and filter data
    filtered_data = validate_and_filter_data(data, train_type)

    # Normalize the data
    normalized_data, median_values_dict = normalize_data(filtered_data, reference_values)

    # Rename the target column to 'label'
    normalized_data.rename(columns={'target': 'label'}, inplace=True)

    # Combine feature names from all dictionaries
    feature_names = list(basic_information.keys()) + list(blood_routine_indicators.keys()) + list(
        blood_biochemical_indicators.keys())

    # Extract features and labels from the normalized_data
    X = normalized_data[feature_names]
    y = normalized_data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, X_test, y_train, y_test), median_values_dict
