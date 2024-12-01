import torch
from torch.utils.data import DataLoader
from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset, create_model, create_tokenizer
from utils.metrics import compute_metrics
import joblib
from transformers import BertForSequenceClassification

def evaluate_model():
    data = load_data(r'C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\combined_data\all_combined_dataset.csv')
    data, label_encoder = preprocess_data(data)
    _, val_texts, _, val_labels = split_data(data)

    tokenizer = create_tokenizer()
    model = BertForSequenceClassification.from_pretrained('savedmodels/mediabias_bert_model')
    model.eval()

    label_encoder = joblib.load('savedmodels/mediabias_bert_model/label_encoder.joblib')

    val_dataset = MediaBiasDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            

    compute_metrics(predictions, true_labels, label_encoder.classes_)

if __name__ == "__main__":
    evaluate_model()
