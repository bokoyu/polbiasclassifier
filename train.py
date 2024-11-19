import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from data.data_loader import load_data, preprocess_data, split_data
from models.bert_classifier import MediaBiasDataset, create_model, create_tokenizer
import os
import joblib

def train_model():
    # Load and preprocess data
    data = load_data(r'C:\Users\Borko\politicalbiasclassifier\data\phrasebias_data\combined_data\all_combined_dataset.csv')
    data, label_encoder = preprocess_data(data)
    train_texts, val_texts, train_labels, val_labels = split_data(data)

    # Initialize tokenizer and datasets
    tokenizer = create_tokenizer()
    train_dataset = MediaBiasDataset(train_texts, train_labels, tokenizer)
    val_dataset = MediaBiasDataset(val_texts, val_labels, tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model()
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

    # Save the model and tokenizer
    save_directory = 'savedmodels/mediabias_bert_model'
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Save the label encoder
    joblib.dump(label_encoder, os.path.join(save_directory, 'label_encoder.joblib'))

if __name__ == "__main__":
    train_model()
