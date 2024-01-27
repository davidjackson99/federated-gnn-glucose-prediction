from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime, timedelta


def preprocess(df, columns_to_scale=['cbg', 'basal']):

    # Handling 'cbg' with mean imputation, respecting 'missing_cbg' column
    cbg_mean = df['cbg'].mean()
    df['cbg'] = df.apply(lambda row: cbg_mean if (row['missing_cbg'] == 1 and pd.isnull(row['cbg'])) else row['cbg'], axis=1)

    start = datetime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    df['5minute_intervals_timestamp'] = df['5minute_intervals_timestamp'].apply(lambda x: start + timedelta(minutes=x)) #convert to date time
    df['5minute_intervals_timestamp'] = df['5minute_intervals_timestamp'].dt.round('1T') #round for uniform distr.
    # Mean imputation for 'basal', 'hr', and 'gsr'
    df['basal'].fillna(df['basal'].mean(), inplace=True)
    df = df.drop(['hr'], axis=1)
    df = df.drop(['missing_cbg'], axis=1)
    df = df.drop(['5minute_intervals_timestamp'], axis=1)
    df['gsr'].fillna(df['gsr'].mean(), inplace=True)

    # Special indicator for 'finger', 'carbInput', and 'bolus'
    df['finger'].fillna(-1, inplace=True)
    df['carbInput'].fillna(-1, inplace=True)
    df['bolus'].fillna(-1, inplace=True)
    # Normalization/Standardization
    if columns_to_scale:
        scaler = StandardScaler()
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df

def get_single_patient_data():

    data_dir = '/Users/davidjackson/Downloads/ohiot1dm'

    # Initialize the result dictionary
    result_dict = {}

    # Loop through each patient
    for patient_number in [570]:
        patient_data = {'train': [], 'test': []}
        # Loop through 'train' and 'test' sets
        for dataset_type in ['train', 'test']:
            csv_filename = f'{patient_number}-ws-{dataset_type}ing_processed.csv'
            csv_path = os.path.join(data_dir, csv_filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
            df = preprocess(df)
            pd.set_option('display.max_columns', None)
            if dataset_type=='train':
                patient_data['train'] = df
            else:
                patient_data['test'] = df
    return patient_data


def get_all_patient_data():

    data_dir = '/Users/davidjackson/Downloads/ohiot1dm'

    # Initialize the result dictionary
    result = []

    # Loop through each patient
    for patient_number in [591, 588, 575, 563, 559, 570, 596, 584, 567,  552, 544, 540]:
        patient_data = {'train': [], 'test': []}
        # Loop through 'train' and 'test' sets
        for dataset_type in ['train', 'test']:
            csv_filename = f'{patient_number}-ws-{dataset_type}ing_processed.csv'
            csv_path = os.path.join(data_dir, csv_filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
            df = preprocess(df)
            pd.set_option('display.max_columns', None)
            if dataset_type=='train':
                patient_data['train'] = df
            else:
                patient_data['test'] = df

        result.append(patient_data)

    return result



def get_all_t_data(training='train'):

    data_dir = '/Users/davidjackson/Downloads/ohiot1dm'

    # Initialize the result dictionary
    result = []

    # Loop through each patient
    for patient_number in [591, 588, 575, 563, 559, 570, 596, 584, 567,  552, 544, 540]:
        patient_data = {'train': [], 'test': []}
        # Loop through 'train' and 'test' sets
        for dataset_type in ['train', 'test']:
            if dataset_type != training:
                continue
            csv_filename = f'{patient_number}-ws-{dataset_type}ing_processed.csv'
            csv_path = os.path.join(data_dir, csv_filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_path)
            df = preprocess(df)
            pd.set_option('display.max_columns', None)
            result.append(df)

    return result
