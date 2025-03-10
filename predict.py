import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

BIAS_MODEL_PATH = "savedmodels/bias_model"
LEANING_MODEL_PATH = "savedmodels/leaning_model"

def load_models():

    bias_tokenizer = BertTokenizer.from_pretrained(BIAS_MODEL_PATH)
    bias_model = BertForSequenceClassification.from_pretrained(BIAS_MODEL_PATH)
    
    try:
        bias_label_encoder = joblib.load(f"{BIAS_MODEL_PATH}/label_encoder.joblib")
    except FileNotFoundError:
        bias_label_encoder = None

    leaning_tokenizer = BertTokenizer.from_pretrained(LEANING_MODEL_PATH)
    leaning_model = BertForSequenceClassification.from_pretrained(LEANING_MODEL_PATH)

    try:
        leaning_label_encoder = joblib.load(f"{LEANING_MODEL_PATH}/label_encoder.joblib")
    except FileNotFoundError:
        leaning_label_encoder = None

    return (bias_tokenizer, bias_model, bias_label_encoder,
            leaning_tokenizer, leaning_model, leaning_label_encoder)

BIAS_TOKENIZER, BIAS_MODEL, BIAS_LABEL_ENCODER, \
LEANING_TOKENIZER, LEANING_MODEL, LEANING_LABEL_ENCODER = load_models()

# cpu/gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BIAS_MODEL.to(DEVICE)
LEANING_MODEL.to(DEVICE)
BIAS_MODEL.eval()
LEANING_MODEL.eval()

def predict_bias(text):
    inputs = BIAS_TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = BIAS_MODEL(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    if BIAS_LABEL_ENCODER is not None:
        predicted_label = BIAS_LABEL_ENCODER.inverse_transform([predicted_class_id])[0]
    else:
        predicted_label = "Neutral" if predicted_class_id == 0 else "Biased"

    print("=== Bias Model ===")
    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probs.cpu().numpy()}")
    print(f"Predicted Label: {predicted_label} | Confidence: {confidence:.4f}\n")

    return predicted_class_id, predicted_label, confidence

def predict_leaning(text):
    inputs = LEANING_TOKENIZER.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = LEANING_MODEL(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)  # shape: (batch_size, #classes)

    predicted_class_id = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()

    if LEANING_LABEL_ENCODER is not None:
        predicted_label = LEANING_LABEL_ENCODER.inverse_transform([predicted_class_id])[0]
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
        # neutral
        return f"[BIAS: {bias_label} ({bias_conf:.2f})]"

