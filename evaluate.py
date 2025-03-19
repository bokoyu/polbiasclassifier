import torch
from torch.utils.data import DataLoader
from models.bert_classifier import MediaBiasDataset, create_tokenizer
from transformers import RobertaForSequenceClassification
from data.data_loader import load_data, preprocess_data, split_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(data_path, do_cleaning=False, cleaning_func=None, batch_size=8):

    df = load_data(data_path)
    df = preprocess_data(df, do_cleaning=do_cleaning, cleaning_func=cleaning_func)

    (X_train, X_val, y_train, y_val,
     X_train_biased, X_val_biased, y_train_biased, y_val_biased) = split_data(df)

    tokenizer = create_tokenizer()

    # Evaluate bias model
    bias_model_path = 'savedmodels/bias_model'
    bias_metrics = evaluate_bias_model(X_val, y_val, tokenizer, bias_model_path, batch_size)
    print("==== BIAS MODEL EVALUATION ====")
    print_metrics(bias_metrics)

    # Evaluate leaning model
    leaning_model_path = 'savedmodels/leaning_model'
    leaning_metrics = evaluate_leaning_model(X_val_biased, y_val_biased, tokenizer, leaning_model_path, batch_size)
    print("==== LEANING MODEL EVALUATION ====")
    print_metrics(leaning_metrics)

def evaluate_bias_model(X_val, y_val, tokenizer, model_path, batch_size=8):
    dataset = MediaBiasDataset(X_val, y_val, tokenizer)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = y_val.tolist()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())

    metrics = compute_classification_metrics(all_labels, all_preds)
    return metrics

def evaluate_leaning_model(X_val, y_val, tokenizer, model_path, batch_size=8):
    dataset = MediaBiasDataset(X_val, y_val, tokenizer)
    val_loader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobertaForSequenceClassification.from_pretrained(model_path)
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

    metrics = compute_classification_metrics(all_labels, all_preds)
    return metrics

def compute_classification_metrics(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average='macro')
    rec = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

def print_metrics(metrics_dict):
    for k, v in metrics_dict.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print()

