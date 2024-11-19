import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification

class MediaBiasDataset(Dataset):
    """Custom Dataset class for Media Bias phrases."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def create_model(num_labels=3):
    """Load a pre-trained BERT model with a classification head."""
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    return model

def create_tokenizer():
    """Load the BERT tokenizer."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer
