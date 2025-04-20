import pandas as pd
import numpy as np
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import argparse

# === Data Loading and Preprocessing ===
def load_data(filepath):
    """Load dataset from CSV or Parquet."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    print(f"Columns in {os.path.basename(filepath)}:", df.columns)
    return df

def clean_text(text):
    """Basic text cleaning."""
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_data(df, text_col='text', bias_col='label', leaning_col='type'):
    """Preprocess data for both bias and leaning models."""
    required_cols = [text_col, bias_col]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}, Found: {df.columns}")
    
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df['bias_label'] = pd.to_numeric(df[bias_col], errors='coerce')
    df = df.dropna(subset=['bias_label', 'cleaned_text'])
    df['bias_label'] = df['bias_label'].astype(int)
    
    if leaning_col in df.columns:
        def map_leaning(bias, leaning):
            if bias == 0 or pd.isna(leaning):
                return np.nan
            leaning = str(leaning).lower()
            if leaning == 'left':
                return 0
            elif leaning == 'right':
                return 1
            elif leaning == 'center':
                return 2
            else:
                return np.nan
        df['leaning_label'] = df.apply(lambda row: map_leaning(row['bias_label'], row[leaning_col]), axis=1)
    else:
        df['leaning_label'] = np.nan
    
    df = df.drop_duplicates(subset='cleaned_text')
    df = df[df['cleaned_text'].str.split().apply(len) > 5]
    
    bias_encoder = LabelEncoder()
    df['bias_encoded'] = bias_encoder.fit_transform(df['bias_label'])
    
    df_biased = df[df['bias_label'] == 1].dropna(subset=['leaning_label'])
    leaning_encoder = LabelEncoder()
    df_biased['leaning_encoded'] = leaning_encoder.fit_transform(df_biased['leaning_label'])
    
    return df, df_biased, bias_encoder, leaning_encoder

# === Model Training ===
def train_bias_model(X_train, y_train, tfidf=None):
    """Train TF-IDF + Logistic Regression for bias detection."""
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
    else:
        X_train_tfidf = tfidf.transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_tfidf, y_train)
    return clf, tfidf

def train_leaning_model(X_train, y_train, tfidf=None):
    """Train TF-IDF + Logistic Regression for leaning detection with SMOTE."""
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
    else:
        X_train_tfidf = tfidf.transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_tfidf_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='multinomial')
    clf.fit(X_train_tfidf_res, y_train_res)
    return clf, tfidf

# === Evaluation ===
def evaluate_model(clf, X_test_tfidf, y_test, label_encoder, model_name="Model"):
    """Evaluate the model and print metrics."""
    y_pred = clf.predict(X_test_tfidf)
    print(f"\n=== {model_name} Evaluation on Test Set ===")
    print("Classification Report:")
    if model_name == "Bias Model":
        target_names = ["Neutral", "Biased"]
    else:
        target_names = ["Left", "Right", "Center"]
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def predict(text, bias_clf, bias_tfidf, leaning_clf, leaning_tfidf, bias_encoder, leaning_encoder):
    """Predict bias and leaning for a given text."""
    cleaned_text = clean_text(text)
    X_tfidf = bias_tfidf.transform([cleaned_text])
    
    bias_pred = bias_clf.predict(X_tfidf)[0]
    bias_prob = bias_clf.predict_proba(X_tfidf)[0].max()
    bias_label = bias_encoder.inverse_transform([bias_pred])[0]
    bias_label_str = "Neutral" if bias_label == 0 else "Biased"
    
    if bias_pred == 1:
        X_leaning_tfidf = leaning_tfidf.transform([cleaned_text])
        leaning_pred = leaning_clf.predict(X_leaning_tfidf)[0]
        leaning_prob = leaning_clf.predict_proba(X_leaning_tfidf)[0].max()
        leaning_label = leaning_encoder.inverse_transform([leaning_pred])[0]
        leaning_label_str = {0: "Left", 1: "Right", 2: "Center"}.get(leaning_label, "Unknown")
        return f"[BIAS: {bias_label_str} ({bias_prob:.2f})] | [LEANING: {leaning_label_str} ({leaning_prob:.2f})]"
    else:
        return f"[BIAS: {bias_label_str} ({bias_prob:.2f})]"

def main(args):
    if not args.train_data or not args.test_data:
        raise ValueError("Please provide both --train-data and --test-data paths")

    df_train = load_data(args.train_data)
    df_train, df_train_biased, bias_encoder, leaning_encoder = preprocess_data(
        df_train, text_col=args.text_col, bias_col=args.bias_col, leaning_col=args.leaning_col
    )

    df_test = load_data(args.test_data)
    df_test, df_test_biased, _, _ = preprocess_data(
        df_test, text_col=args.text_col, bias_col=args.bias_col, leaning_col=args.leaning_col
    )

    X_train = df_train['cleaned_text'].values
    y_train_bias = df_train['bias_encoded'].values
    X_train_biased = df_train_biased['cleaned_text'].values
    y_train_leaning = df_train_biased['leaning_encoded'].values

    X_test = df_test['cleaned_text'].values
    y_test_bias = df_test['bias_encoded'].values
    X_test_biased = df_test_biased['cleaned_text'].values
    y_test_leaning = df_test_biased['leaning_encoded'].values

    bias_clf, bias_tfidf = train_bias_model(X_train, y_train_bias)
    leaning_clf, leaning_tfidf = train_leaning_model(X_train_biased, y_train_leaning)

    X_test_tfidf = bias_tfidf.transform(X_test)
    evaluate_model(bias_clf, X_test_tfidf, y_test_bias, bias_encoder, "Bias Model")
    
    X_test_biased_tfidf = leaning_tfidf.transform(X_test_biased)
    evaluate_model(leaning_clf, X_test_biased_tfidf, y_test_leaning, leaning_encoder, "Leaning Model")

    os.makedirs("tfidf_models", exist_ok=True)
    joblib.dump(bias_tfidf, 'tfidf_models/tfidf_bias_vectorizer.joblib')
    joblib.dump(bias_clf, 'tfidf_models/tfidf_bias_model.joblib')
    joblib.dump(bias_encoder, 'tfidf_models/tfidf_bias_encoder.joblib')
    joblib.dump(leaning_tfidf, 'tfidf_models/tfidf_leaning_vectorizer.joblib')
    joblib.dump(leaning_clf, 'tfidf_models/tfidf_leaning_model.joblib')
    joblib.dump(leaning_encoder, 'tfidf_models/tfidf_leaning_encoder.joblib')

    if args.text:
        result = predict(args.text, bias_clf, bias_tfidf, leaning_clf, leaning_tfidf, bias_encoder, leaning_encoder)
        print(f"\nPrediction for '{args.text}':\n{result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF Political Bias and Leaning Classifier with Train/Test Split")
    parser.add_argument('--train-data', type=str, help="Path to the training dataset (CSV or Parquet)")
    parser.add_argument('--test-data', type=str, help="Path to the testing dataset (CSV or Parquet)")
    parser.add_argument('--text', type=str, help="Text to predict bias and leaning for")
    parser.add_argument('--text-col', type=str, default='text', help="Column name for text data")
    parser.add_argument('--bias-col', type=str, default='label', help="Column name for bias labels")
    parser.add_argument('--leaning-col', type=str, default='type', help="Column name for leaning labels")
    
    args = parser.parse_args()
    main(args)