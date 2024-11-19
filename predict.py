import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

def predict_bias(text):
    tokenizer = BertTokenizer.from_pretrained('savedmodels/mediabias_bert_model')
    model = BertForSequenceClassification.from_pretrained('savedmodels/mediabias_bert_model')
    label_encoder = joblib.load('savedmodels/mediabias_bert_model/label_encoder.joblib')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        truncation=True,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits

        temperature = 2.0
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)

        predicted_class_id = torch.argmax(probs, dim=1).item()

    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")

    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label

if __name__ == "__main__":
    sample_text = "Universities need affirmative action policies to promote diversity."
    bias = predict_bias(sample_text)
    print(f"Predicted Bias: {bias}")
