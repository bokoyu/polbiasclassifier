import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup, pipeline
from tqdm import tqdm
import shutil
import torch.nn as nn
import numpy as np
import re
import contractions
from deep_translator import GoogleTranslator
import time
import pandas as pd

from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset, create_model, create_tokenizer

TRANSLATORS = {
    'fr': {
        'en_to': pipeline('translation_en_to_fr', model='Helsinki-NLP/opus-mt-en-fr'),
        'to_en': pipeline('translation_fr_to_en', model='Helsinki-NLP/opus-mt-fr-en')
    },
    'de': {
        'en_to': pipeline('translation_en_to_de', model='Helsinki-NLP/opus-mt-en-de'),
        'to_en': pipeline('translation_de_to_en', model='Helsinki-NLP/opus-mt-de-en')
    }
}

def custom_clean(text):
    print("cleaning text")
    text = re.sub(r'\d+', '', text)        # remove numbers
    text = contractions.fix(text)          # expand contractions
    text = text.strip().lower()
    return text

def back_translate(text: str, mid_lang: str = 'fr') -> str:
    try:
        translated = GoogleTranslator(source='en', target=mid_lang).translate(text)
        back_translated = GoogleTranslator(source=mid_lang, target='en').translate(translated)
        return back_translated
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def augment_center_samples(df: pd.DataFrame, num_augments: int = 3) -> pd.DataFrame:
    if 'type' not in df.columns:
        return df
        
    center_samples = df[df['type'] == 'center'].copy()
    if len(center_samples) == 0:
        return df

    augmented_rows = []
    for _, row in center_samples.iterrows():
        for _ in range(num_augments):
            lang = np.random.choice(['fr', 'de', 'es', 'it'])
            new_text = back_translate(row['text'], mid_lang=lang)
            new_row = row.copy()
            new_row['text'] = new_text
            augmented_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

def train_model(data_path, do_cleaning=True, cleaning_func=custom_clean,
                epochs=3, batch_size=8, overwrite=False):
    df = load_data(data_path)
    df = augment_center_samples(df, num_augments=3)
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
        save_path=bias_model_path,
        epochs=epochs,
        batch_size=batch_size
    )

    train_leaning_model(
        X_train_biased, y_train_biased, X_val_biased, y_val_biased, tokenizer,
        save_path=leaning_model_path,
        epochs=epochs,
        batch_size=batch_size
    )


def train_bias_model(X_train, y_train, X_val, y_val, tokenizer,
                     save_path, epochs=3, batch_size=8,
                     lr=3e-5, weight_decay=0.01,
                     dropout_prob=0.3, patience=2):

    train_dataset = MediaBiasDataset(X_train, y_train, tokenizer)
    val_dataset   = MediaBiasDataset(X_val,   y_val,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    print("CUDA available?", torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = create_model(num_labels=2)
    model.config.hidden_dropout_prob = dropout_prob
    model.config.attention_probs_dropout_prob = dropout_prob
    model.to(device)

    # === Class Weights for Imbalance ===
    label_counts = np.bincount(y_train)
    total = sum(label_counts)
    class_weights = [total / c if c != 0 else 0.0 for c in label_counts]

    print("Class counts (Bias Model): ", label_counts)
    print("Class weights (Bias Model): ", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # === Learning Rate Scheduler ===
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # === Early Stopping Setup ===
    best_val_loss = float('inf')
    no_improve_count = 0

    # === Training Loop ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Bias Model Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} [Bias] - Training Loss: {avg_loss:.4f}")

        val_loss = evaluate_on_loader(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} [Bias] - Validation Loss: {val_loss:.4f}")

        # === Early Stopping Check ===
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0

            # Save the best model so far
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  -> New best model saved at epoch {epoch+1} (val_loss={val_loss:.4f})")

        else:
            no_improve_count += 1
            print(f"  -> No improvement. Patience {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print("Early stopping triggered!")
                break

    print("Bias model training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


def train_leaning_model(X_train, y_train, X_val, y_val, tokenizer,
                        save_path, epochs=3, batch_size=8,
                        lr=2e-5, weight_decay=0.01,
                        dropout_prob=0.3, patience=2):

    train_dataset = MediaBiasDataset(X_train, y_train, tokenizer)
    val_dataset   = MediaBiasDataset(X_val,   y_val,   tokenizer)

    class_sample_counts = np.bincount(y_train)
    weights = 1. / class_sample_counts
    sample_weights = np.array([weights[label] for label in y_train])

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_labels=3)  # 0=Left, 1=Right, 2=Center

    model.config.hidden_dropout_prob = dropout_prob
    model.config.attention_probs_dropout_prob = dropout_prob
    model.to(device)

    total = sum(class_sample_counts)
    class_weights = [total / c if c != 0 else 0.0 for c in class_sample_counts]

    print("Class counts (Leaning Model): ", class_sample_counts)
    print("Class weights (Leaning Model): ", class_weights)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Early stopping
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Leaning Model Epoch {epoch+1}")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} [Leaning] - Training Loss: {avg_loss:.4f}")

        # Validate
        val_loss = evaluate_on_loader(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{epochs} [Leaning] - Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0

            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"  -> New best Leaning model saved at epoch {epoch+1} (val_loss={val_loss:.4f})")
        else:
            no_improve_count += 1
            print(f"  -> No improvement. Patience {no_improve_count}/{patience}")
            if no_improve_count >= patience:
                print("Early stopping triggered!")
                break

    print("Leaning model training complete.")
    print(f"Best val loss: {best_val_loss:.4f}")


def evaluate_on_loader(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(data_loader)
