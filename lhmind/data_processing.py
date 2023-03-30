# lhmind/data_processing.py
from sklearn.model_selection import train_test_split

def normalize_data(data, reference_values):
    normalized_data = data.copy()

    # Normalize age
    normalized_data['AGE'] = normalized_data['AGE'].apply(lambda x: (x - 1) / (100 - 1))

    # Normalize blood test data based on gender and reference values
    for _, row in reference_values.iterrows():
        gender, indicator, min_val, max_val = row['gender'], row['indicator'], row['min_val'], row['max_val']
        mask = normalized_data['gender'] == gender
        normalized_data.loc[mask, indicator] = (normalized_data.loc[mask, indicator] - min_val) / (max_val - min_val)

    return normalized_data

def preprocess_data(data, train_type, reference_values):
    # Normalize the data
    normalized_data = normalize_data(data, reference_values)

    # Set the target column based on the train_type
    if train_type == 'normal_cancer':
        target_col = 'cancer_label'
    elif train_type == 'normal_inflam':
        target_col = 'inflam_label'
    elif train_type == 'inflam_cancer':
        target_col = 'inflam_cancer_label'
    else:
        raise ValueError(f"Invalid train_type: {train_type}")

    # Split the data into features and target
    X = normalized_data.drop(columns=[target_col])
    y = data[target_col]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
