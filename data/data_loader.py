import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(file_path):
    #load dataset
    data = pd.read_csv(file_path)

    if 'PHRASE' not in data.columns or 'calculated_bias' not in data.columns:
        raise ValueError("Expected columns 'PHRASE' and 'calculated_bias' not found in dataset. Please check the column names.")

    data = data[['PHRASE', 'calculated_bias']]
    return data

def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['bias_label'] = label_encoder.fit_transform(data['calculated_bias'])
    return data, label_encoder

def split_data(data, test_size=0.2):
    return train_test_split(
        data['PHRASE'],
        data['bias_label'],
        test_size=test_size,
        random_state=42
    )