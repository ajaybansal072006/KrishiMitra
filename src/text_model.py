from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf

def build_text_feature_extractor():
    """
    Extracts text embeddings using DistilBERT (not classification head).
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def extract_text_features(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
    return cls_embedding

def analyze_text(text):
    """
    Analyze text and return extracted features or summary info.
    """
    model, tokenizer = build_text_feature_extractor()
    features = extract_text_features(model, tokenizer, text)

    # You can later replace this with classification logic or sentiment analysis
    return {
        "features": features.numpy().tolist(),   # Convert tensor to list for Streamlit
        "summary": f"Processed text: {text[:80]}..."
    }
