import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import re

def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Columns in the dataset:", df.columns)
    return df

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_data(df, text_col='PHRASE', bias_col='calculated_bias'):
    if bias_col not in df.columns:
        raise ValueError(f"'{bias_col}' column not found in the DataFrame.")
    df[bias_col] = pd.to_numeric(df[bias_col], errors='coerce')
    df = df.dropna(subset=[bias_col])

    def map_bias(value):
        if value > 0:
            return 'Right'
        elif value < 0:
            return 'Left'
        else:
            return 'Neutral'

    df['label'] = df[bias_col].apply(map_bias)
    df['cleaned_text'] = df[text_col].apply(clean_text)

    df = df.drop_duplicates(subset='cleaned_text')
    df = df[df['cleaned_text'].str.split().apply(len) > 5]

    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])

    return df, label_encoder

def main():
    data_path = r'C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\combined_data\all_combined_dataset.csv'
    df = load_data(data_path)

    df, label_encoder = preprocess_data(df, text_col='PHRASE', bias_col='calculated_bias')

    X = df['cleaned_text'].values
    y = df['encoded_label'].values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_val_tfidf)

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    joblib.dump(clf, 'tfidf_logreg_model.joblib')
    joblib.dump(label_encoder, 'tfidf_label_encoder.joblib')

if __name__ == '__main__':
    main()
