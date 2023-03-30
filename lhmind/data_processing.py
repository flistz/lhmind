# lhmind/data_processing.py
from sklearn.model_selection import train_test_split
from .constants import basic_information, blood_routine_indicators, blood_biochemical_indicators


def normalize_data(data, reference_values):
    normalized_data = data.copy()

    # Normalize age
    normalized_data['AGE'] = normalized_data['AGE'].apply(lambda x: (x - 1) / (100 - 1))

    # Combine feature names from both dictionaries
    feature_names = list(blood_routine_indicators.keys()) + list(blood_biochemical_indicators.keys())

    # Normalize blood test data based on gender and reference values
    for gender, group_data in normalized_data.groupby('SEX'):
        if gender in ['M', 'F']:
            for feature_name in feature_names:
                # Get the reference values for the current feature and gender
                ref_row = reference_values.loc[(reference_values['indicator'] == feature_name) & (reference_values['gender'] == gender)]
                min_val, max_val = ref_row['min'].values[0], ref_row['max'].values[0]

                # Handle missing data by filling with the median value for the current gender group
                median_val = group_data[feature_name].median()
                normalized_data.loc[normalized_data['SEX'] == gender, feature_name] = normalized_data.loc[normalized_data['SEX'] == gender, feature_name].fillna(median_val)

                # Normalize the data in-place for the current feature and gender
                mask = normalized_data['SEX'] == gender
                normalized_data.loc[mask, feature_name] = (normalized_data.loc[mask, feature_name] - min_val) / (max_val - min_val)

    return normalized_data

def filter_and_map_data(normalized_data, target_col):
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
    filtered_data = normalized_data[normalized_data['DISEASE'].apply(filter_func)]

    # Map the DISEASE column values based on the target_col
    filtered_data_with_label = filtered_data.copy()
    filtered_data_with_label['label'] = filtered_data['DISEASE'].apply(map_func)

    value_counts = filtered_data_with_label['label'].value_counts()

    if not bool(len(value_counts) == 2 and 0 in value_counts.index and 1 in value_counts.index):
        raise Exception('数据的训练类型与数据的标签类型数量不符')

    return filtered_data_with_label


def preprocess_data(data, train_type, reference_values):
    # Normalize the data
    normalized_data = normalize_data(data, reference_values)

    # validate train_type
    if train_type not in ['normal_cancer', 'normal_inflam', 'inflam_cancer', 'normal_inflam&cancer']:
        raise ValueError(f"Invalid train_type: {train_type}")

    # Filter and map data based on the train_type
    filtered_mapped_data = filter_and_map_data(normalized_data, train_type)

    # Rename the target column to 'label'
    filtered_mapped_data.rename(columns={'target': 'label'}, inplace=True)

    # Combine feature names from all dictionaries
    feature_names = list(basic_information.keys()) + list(blood_routine_indicators.keys()) + list(blood_biochemical_indicators.keys())

    # Extract features and labels from the filtered_mapped_data
    X = filtered_mapped_data[feature_names]
    y = filtered_mapped_data['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
