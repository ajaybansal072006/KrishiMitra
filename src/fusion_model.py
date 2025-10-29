import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import os

# -------------------------------------------------------
# 🧠 Build Fusion Model (Learnable)
# -------------------------------------------------------
def build_fusion_model(image_feature_dim=256, text_feature_dim=768, num_classes=4):
    """
    Builds a fusion model combining image and text embeddings.
    """
    image_input = Input(shape=(image_feature_dim,), name="image_features")
    text_input = Input(shape=(text_feature_dim,), name="text_features")

    # Fusion
    x = layers.Concatenate()([image_input, text_input])
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name="fusion_output")(x)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -------------------------------------------------------
# 🔮 Fused Prediction Function
# -------------------------------------------------------
def fuse_predictions(image_features, text_features, weights_path="models/fusion_model_weights.h5"):
    """
    Combines image and text features to predict disease.
    Requires a pretrained fusion model (weights file).
    """
    # Load the model
    model = build_fusion_model()
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("✅ Loaded fusion model weights.")
    else:
        print("⚠️ Warning: Using untrained fusion model (random predictions).")

    # Convert to numpy arrays
    image_features = np.array(image_features)
    text_features = np.array(text_features)

    # Remove extra batch dimension if necessary
    if image_features.ndim == 3:
        image_features = np.squeeze(image_features, axis=0)
    if text_features.ndim == 3:
        text_features = np.squeeze(text_features, axis=0)

    # Ensure batch dimension
    if image_features.ndim == 1:
        image_features = np.expand_dims(image_features, axis=0)
    if text_features.ndim == 1:
        text_features = np.expand_dims(text_features, axis=0)

    # Sanity check shapes
    print("🖼️ Image features shape:", image_features.shape)
    print("💬 Text features shape:", text_features.shape)

    # Predict
    preds = model.predict([image_features, text_features])
    class_idx = np.argmax(preds, axis=1)[0]

    # Ensure consistent class labels (same as image_model)
    class_labels = ["Leaf_Blight", "Rust", "Healthy", "Bacterial_Spot"]
    predicted_label = class_labels[class_idx] if class_idx < len(class_labels) else "Unknown"

    confidence = float(np.max(preds))

    return predicted_label, confidence, preds.tolist()
