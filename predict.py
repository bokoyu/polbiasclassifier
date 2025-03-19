import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import joblib

BIAS_MODEL_PATH = "savedmodels/bias_model"
LEANING_MODEL_PATH = "savedmodels/leaning_model"

def load_bias_components():
    """Load bias model components only when needed"""
    tokenizer = RobertaTokenizer.from_pretrained(BIAS_MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(BIAS_MODEL_PATH)
    try:
        label_encoder = joblib.load(f"{BIAS_MODEL_PATH}/label_encoder.joblib")
    except FileNotFoundError:
        label_encoder = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, label_encoder, device

def load_leaning_components():
    """Load leaning model components only when needed"""
    tokenizer = RobertaTokenizer.from_pretrained(LEANING_MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(LEANING_MODEL_PATH)
    try:
        label_encoder = joblib.load(f"{LEANING_MODEL_PATH}/label_encoder.joblib")
    except FileNotFoundError:
        label_encoder = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return tokenizer, model, label_encoder, device

def predict_bias(text):
    # Load components on demand
    tokenizer, model, label_encoder, device = load_bias_components()
    
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    else:
        predicted_label = "Neutral" if predicted_class_id == 0 else "Biased"

    print("=== Bias Model ===")
    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    print(f"Predicted Label: {predicted_label} | Confidence: {confidence:.4f}\n")

    return predicted_class_id, predicted_label, confidence

def predict_leaning(text):
    # Load components on demand
    tokenizer, model, label_encoder, device = load_leaning_components()
    
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    if label_encoder is not None:
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    else:
        leaning_map = {0: "Left", 1: "Right", 2: "Center"}
        predicted_label = leaning_map.get(predicted_class_id, "Unknown")

    print("=== Leaning Model ===")
    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    print(f"Predicted Leaning: {predicted_label} | Confidence: {confidence:.4f}\n")

    return predicted_class_id, predicted_label, confidence

def predict(text):
    bias_id, bias_label, bias_conf = predict_bias(text)

    if bias_id == 1:
        lean_id, lean_label, lean_conf = predict_leaning(text)
        return f"[BIAS: {bias_label} ({bias_conf:.2f})] | [LEANING: {lean_label} ({lean_conf:.2f})]"
    else:
        return f"[BIAS: {bias_label} ({bias_conf:.2f})]"