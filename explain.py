import shap
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIAS_MODEL_PATH = "savedmodels/bias_model"
LEANING_MODEL_PATH = "savedmodels/leaning_model"

def load_bias_model():
    t = BertTokenizer.from_pretrained(BIAS_MODEL_PATH)
    m = BertForSequenceClassification.from_pretrained(BIAS_MODEL_PATH)
    m.to(DEVICE)
    m.eval()
    return t, m

def load_leaning_model():
    t = BertTokenizer.from_pretrained(LEANING_MODEL_PATH)
    m = BertForSequenceClassification.from_pretrained(LEANING_MODEL_PATH)
    m.to(DEVICE)
    m.eval()
    return t, m

def predict_bias_proba(texts, tokenizer, model, max_length=512):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    for k, v in enc.items():
        enc[k] = v.to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1).cpu().numpy()

def predict_leaning_proba(texts, tokenizer, model, max_length=512):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    for k, v in enc.items():
        enc[k] = v.to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1).cpu().numpy()

def kernel_shap_explain_bias(text_list, background_texts=None, max_length=512):
    bt, bm = load_bias_model()
    def fn(x):
        # x is shape (N,1) of strings -> flatten -> pass to model
        texts = [row[0] for row in x]
        return predict_bias_proba(texts, bt, bm, max_length)
    if background_texts is None:
        background_texts = ["Neutral sample text", "Another short neutral example"]
    background_data = np.array([[s] for s in background_texts], dtype=object)
    explainer = shap.KernelExplainer(fn, background_data)
    test_data = np.array([[s] for s in text_list], dtype=object)
    shap_values = explainer.shap_values(test_data, nsamples=50)
    return shap_values

def kernel_shap_explain_leaning(text_list, background_texts=None, max_length=512):
    tk, md = load_leaning_model()
    def fn(x):
        texts = [row[0] for row in x]
        return predict_leaning_proba(texts, tk, md, max_length)
    if background_texts is None:
        background_texts = ["Generic policy statement", "Some random text"]
    background_data = np.array([[s] for s in background_texts], dtype=object)
    explainer = shap.KernelExplainer(fn, background_data)
    test_data = np.array([[s] for s in text_list], dtype=object)
    shap_values = explainer.shap_values(test_data, nsamples=50)
    return shap_values

def lime_explain_bias(text, class_names=["Neutral","Biased"], max_length=512):
    tk, md = load_bias_model()
    def predict_proba(txts):
        return predict_bias_proba(txts, tk, md, max_length)
    e = LimeTextExplainer(class_names=class_names)
    return e.explain_instance(text, predict_proba, num_features=10)

def lime_explain_leaning(text, class_names=["Left","Right"], max_length=512):
    tk, md = load_leaning_model()
    def predict_proba(txts):
        return predict_leaning_proba(txts, tk, md, max_length)
    e = LimeTextExplainer(class_names=class_names)
    return e.explain_instance(text, predict_proba, num_features=10)
