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
import time
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from llm_augmentation import augment_with_t5

from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset, create_model, create_tokenizer

DEVICE = 0 if torch.cuda.is_available() else -1
DEFAULT = "data/babe/train-00000-of-00001.parquet"

TRANSLATORS = {
    'fr': {
        'en_to': pipeline('translation_en_to_fr', 
                         model='Helsinki-NLP/opus-mt-en-fr',
                         device=DEVICE,
                         torch_dtype=torch.float16 if DEVICE >=0 else torch.float32),
        'to_en': pipeline('translation_fr_to_en',
                         model='Helsinki-NLP/opus-mt-fr-en',
                         device=DEVICE,
                         torch_dtype=torch.float16 if DEVICE >=0 else torch.float32)
    },
    'de': {
        'en_to': pipeline('translation_en_to_de', 
                         model='Helsinki-NLP/opus-mt-en-de',
                         device=DEVICE,
                         torch_dtype=torch.float16 if DEVICE >=0 else torch.float32),
        'to_en': pipeline('translation_de_to_en',
                         model='Helsinki-NLP/opus-mt-de-en',
                         device=DEVICE,
                         torch_dtype=torch.float16 if DEVICE >=0 else torch.float32)
    }
}

def custom_clean(text):
    print("cleaning text")
    text = re.sub(r'\d+', '', text)        # remove numbers
    text = contractions.fix(text)          # expand contractions
    text = text.strip().lower()
    return text

@lru_cache(maxsize=5000)
def back_translate(text: str, mid_lang: str) -> str:
    try:
        translated = TRANSLATORS[mid_lang]['en_to'](text, max_length=512)[0]['translation_text']
        back_translated = TRANSLATORS[mid_lang]['to_en'](translated, max_length=512)[0]['translation_text']
        return back_translated
    except Exception as e:
        print(f"Translation failed for '{text[:50]}...': {e}")
        return text

def augment_center_samples(df: pd.DataFrame, num_augments: int = 2) -> pd.DataFrame:
    if 'type' not in df.columns or df[df['type'] == 'center'].empty:
        return df

    center_samples = df[df['type'] == 'center'].copy()
    augmented_rows = []

    def process_row(row):
        lang = np.random.choice(['fr', 'de'])
        new_row = row.copy()
        new_row['text'] = back_translate(row['text'], lang)
        return new_row

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for _ in range(num_augments):
            for _, row in center_samples.iterrows():
                futures.append(executor.submit(process_row, row))
        
        # progress bar
        for future in tqdm(futures, desc="Augmenting Center Samples", unit="text"):
            augmented_rows.append(future.result())

    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

def train_model(
    data_path: str | None=None,
    do_cleaning=True,
    cleaning_func=custom_clean,
    epochs=3,
    batch_size=8,
    overwrite=False,
    lr_bias=3e-5, 
    lr_lean=2e-5    
):
    df = load_data(data_path or DEFAULT)
    df = augment_with_t5(df, num_augments=1)
    df = augment_center_samples(df, num_augments=2)

    print("===== Label Distribution =====")
    print(df['label'].value_counts(dropna=False))

    if 'type' in df.columns:
        print("\n===== Type Distribution (Among Biased) =====")
        print(df[df['label'] == 1]['type'].value_counts(dropna=False))

    df = preprocess_data(df, do_cleaning=do_cleaning, cleaning_func=cleaning_func)

    (
        X_train, X_val, y_train, y_val,
        X_train_biased, X_val_biased,
        y_train_biased, y_val_biased
    ) = split_data(df)

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
        batch_size=batch_size,
        lr=lr_bias             
    )

    train_leaning_model(
        X_train_biased, y_train_biased, X_val_biased, y_val_biased, tokenizer,
        save_path=leaning_model_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr_lean             
    )



def train_bias_model(X_train, y_train, X_val, y_val, tokenizer,
                     save_path, epochs=3, batch_size=8,
                     lr=None, weight_decay=0.01,
                     dropout_prob=0.3, patience=2):
    
    if lr is None:
        lr = 3e-5

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
                        lr=None, weight_decay=0.01,
                        dropout_prob=0.3, patience=2):

    train_dataset = MediaBiasDataset(X_train, y_train, tokenizer)
    val_dataset   = MediaBiasDataset(X_val,   y_val,   tokenizer)

    if lr is None:
        lr = 2e-5
    # n_classes = 3
    # y_train = y_train.astype(int)
    # class_sample_counts = np.bincount(y_train, minlength=n_classes)

    # class_sample_counts = np.where(class_sample_counts == 0, 1, class_sample_counts)


    # weights = 1.0 / class_sample_counts
    # sample_weights = np.array([weights[label] for label in y_train])

    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(num_labels=3)  # 0=Left, 1=Right, 2=Center

    model.config.hidden_dropout_prob = dropout_prob
    model.config.attention_probs_dropout_prob = dropout_prob
    model.to(device)

    # total = sum(class_sample_counts)
    # class_weights = [total / c if c != 0 else 0.0 for c in class_sample_counts]

    class_weights = [1.0, 1.0, 1.7]

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
