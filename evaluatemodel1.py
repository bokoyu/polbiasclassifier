import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset

def evaluate_bias_model(data_path, batch_size=8):
    print("Loading dataset for bias model evaluation...")
    df = load_data(data_path)
    
    # Step 1 evaluation (Bias vs. Neutral)
    X_val, y_val = df['text'], df['label']

    tokenizer = BertTokenizer.from_pretrained("savedmodels/bias_model")
    model = BertForSequenceClassification.from_pretrained("savedmodels/bias_model")

    dataset = MediaBiasDataset(X_val, y_val, tokenizer)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = y_val.tolist()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    print("\n==== BIAS MODEL EVALUATION ====")
    print(classification_report(all_labels, all_preds, target_names=["Neutral", "Biased"]))
    
    return accuracy_score(all_labels, all_preds)

if __name__ == "__main__":
    data_path = "data/babe/test-00000-of-00001.parquet"
    evaluate_bias_model(data_path)