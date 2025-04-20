import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialise T5 model and tokenizer (load once globally to avoid repeated loading)
T5_TOKENIZER = T5Tokenizer.from_pretrained("t5-base") 
T5_MODEL = T5ForConditionalGeneration.from_pretrained("t5-base")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T5_MODEL.to(DEVICE)
T5_MODEL.eval()

def llm_t5_paraphrase(text: str, label: int, type_val: str = None, max_length: int = 512) -> str:

    if label == 0:
        prompt = f"paraphrase this factual statement: {text}"
    else:
        type_str = type_val if type_val else "neutral"
        prompt = f"paraphrase this {type_str}-leaning opinion: {text}"

    inputs = T5_TOKENIZER(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = T5_MODEL.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4, 
            early_stopping=True,
            no_repeat_ngram_size=2 
        )

    paraphrased_text = T5_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    
    if paraphrased_text.startswith(prompt):
        paraphrased_text = paraphrased_text[len(prompt):].strip()
    
    return paraphrased_text

def augment_with_t5(df: pd.DataFrame, num_augments: int = 1) -> pd.DataFrame:
    augmented_rows = []

    def process_row(row):
        augments = []
        text, label, type_val = row['text'], row['label'], row.get('type', None)
        for _ in range(num_augments):
            try:
                aug_text = llm_t5_paraphrase(text, label, type_val)
                augments.append(row.copy().to_dict() | {'text': aug_text})
            except Exception as e:
                print(f"Failed to paraphrase '{text[:50]}...': {e}")
                augments.append(row.copy().to_dict())  # Keep original on failure
        return augments

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(executor.submit(process_row, row))
        
        for future in tqdm(futures, desc="Augmenting with T5", unit="text"):
            augmented_rows.extend(future.result())

    return pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)