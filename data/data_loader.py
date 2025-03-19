import pandas as pd
from sklearn.model_selection import train_test_split
import os

def map_type_to_leaning(label_val, type_val):
    if pd.isnull(type_val):
        type_val = 'null'

    if label_val == 0 or type_val == 'null':
        return None 
    elif type_val == 'left':
        return 0
    elif type_val == 'right':
        return 1
    elif type_val == 'center':
        return 2
    else:
        raise ValueError(f"Unrecognised combination: label={label_val}, type={type_val}")

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    

    df = pd.read_parquet(file_path)

    required_columns = ['text', 'label', 'type']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if 'biased_words' in df.columns:
        df['augmented_text'] = df.apply(_merge_biased_words, axis=1)
        text_column = 'augmented_text'
    else:
        text_column = 'text'

    df['leaning'] = df.apply(
        lambda row: map_type_to_leaning(row['label'], row['type']),
        axis=1
    )

    df.dropna(subset=[text_column, 'label'], inplace=True)
    df.drop_duplicates(subset=[text_column, 'label', 'type'], inplace=True)

    df['final_text'] = df[text_column]

    return df

def _merge_biased_words(row):
    txt = str(row['text'])
    biased_list = row['biased_words']
    if not isinstance(biased_list, list):
        return txt
    joined = " ".join(biased_list)
    return f"{txt} [BIASWORDS] {joined}"

def preprocess_data(df, do_cleaning=False, cleaning_func=None):
    if do_cleaning and cleaning_func is not None:
        df['final_text'] = df['final_text'].astype(str).apply(cleaning_func)
    
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df['final_text']
    y = df['label']  #label

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    df_biased = df[df['label'] == 1].copy()
    df_biased = df_biased.dropna(subset=['leaning'])
    X_biased = df_biased['final_text']
    y_biased = df_biased['leaning']  # left=0, right=1, center=2

    X_train_biased, X_val_biased, y_train_biased, y_val_biased = train_test_split(
        X_biased, y_biased, test_size=test_size, random_state=random_state, stratify=y_biased
    )

    return (X_train, X_val, y_train, y_val,
            X_train_biased, X_val_biased, y_train_biased, y_val_biased)
