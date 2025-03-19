import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import shutil
import torch.nn as nn
import numpy as np

from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset, create_model, create_tokenizer


def train_model(data_path, do_cleaning=False, cleaning_func=None, epochs=3, batch_size=8, overwrite=False):
    df = load_data(data_path)
    print("===== Label Distribution =====")
    print(df['label'].value_counts(dropna=False))

    if 'type' in df.columns:
        print("\n===== Type Distribution (Among Biased) =====")
        print(df[df['label'] == 1]['type'].value_counts(dropna=False))

    df = preprocess_data(df, do_cleaning=do_cleaning, cleaning_func=cleaning_func)

    (X_train, X_val, y_train, y_val,
     X_train_biased, X_val_biased, y_train_biased, y_val_biased) = split_data(df)
    tokenizer = create_tokenizer()

    bias_model_path = 'savedmodels/bias_model'
    leaning_model_path = 'savedmodels/leaning_model'


    if overwrite and os.path.exists(bias_model_path):
        shutil.rmtree(bias_model_path)
    if overwrite and os.path.exists(leaning_model_path):
        shutil.rmtree(leaning_model_path)

    train_bias_model(
        X_train, y_train, X_val, y_val, tokenizer,
        bias_model_path, epochs=epochs, batch_size=batch_size
    )


    train_leaning_model(
        X_train_biased, y_train_biased, X_val_biased, y_val_biased, tokenizer,
        leaning_model_path, epochs=epochs, batch_size=batch_size
    )
 
def train_bias_model(X_train, y_train, X_val, y_val, tokenizer,
                     save_path, epochs=3, batch_size=8):

    train_dataset = MediaBiasDataset(X_train, y_train, tokenizer)
    val_dataset = MediaBiasDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_labels=2) 
    model.to(device)

    # compute class weights

    label_counts = np.bincount(y_train)
    total = sum(label_counts)
    class_weights = [total / c if c != 0 else 0.0 for c in label_counts]

    print("Class counts (Bias Model): ", label_counts)
    print("Class weights (Bias Model): ", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Bias Model Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} [Bias] - Loss: {avg_loss:.4f}")

        # Evaluate on val
        val_loss = evaluate_on_loader(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} [Bias] - Val Loss: {val_loss:.4f}")

    # Save
    os.makedirs(save_path, exist_ok=True)
    print(f"Created directory (if needed) at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Bias model saved to {save_path}")

def train_leaning_model(X_train, y_train, X_val, y_val, tokenizer,
                        save_path, epochs=3, batch_size=8):

    train_dataset = MediaBiasDataset(X_train, y_train, tokenizer)
    val_dataset = MediaBiasDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_labels=3)  # 0=Left, 1=Right, 2=center
    model.to(device)

    label_counts = np.bincount(y_train)
    total = sum(label_counts)
    class_weights = [total / c if c != 0 else 0.0 for c in label_counts]

    print("Class counts (Bias Model): ", label_counts)
    print("Class weights (Bias Model): ", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Leaning Model Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} [Leaning] - Loss: {avg_loss:.4f}")

        # Evaluate on val
        val_loss = evaluate_on_loader(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} [Leaning] - Val Loss: {val_loss:.4f}")

    # Save
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Leaning model saved to {save_path}")

def evaluate_on_loader(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)
