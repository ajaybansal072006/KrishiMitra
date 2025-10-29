from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import numpy as np

# -------------------------------------------------------
# 🧠 Load DistilBERT once
# -------------------------------------------------------
_tokenizer = None
_model = None

def build_text_feature_extractor():
    """
    Loads the DistilBERT model and tokenizer (only once for efficiency).
    """
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    return _model, _tokenizer


# -------------------------------------------------------
# 🔡 Extract Text Features
# -------------------------------------------------------
def extract_text_features(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding (1, 768)
    return cls_embedding


# -------------------------------------------------------
# 🧩 Analyze Text Description
# -------------------------------------------------------
def analyze_text(text):
    """
    Converts text description into a feature vector and predicts a rough disease label.
    """
    model, tokenizer = build_text_feature_extractor()
    features = extract_text_features(model, tokenizer, text).numpy()  # shape (1, 768)

    # Example heuristic: classify text based on keywords (for interpretability)
    text_lower = text.lower()
    if any(k in text_lower for k in ["yellow", "spot", "hole", "bacteria"]):
        label = "Bacterial_Spot"
    elif any(k in text_lower for k in ["rust", "brown", "powder"]):
        label = "Rust"
    elif any(k in text_lower for k in ["dry", "blight", "burn", "wilt"]):
        label = "Leaf_Blight"
    elif any(k in text_lower for k in ["healthy", "green", "normal"]):
        label = "Healthy"
    else:
        label = "Unknown"

    return label, features
