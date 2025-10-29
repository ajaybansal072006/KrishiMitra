import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np

def build_fusion_model(image_feature_dim=256, text_feature_dim=768, num_classes=4):
    """
    Learnable fusion model combining image and text features.
    """
    image_input = Input(shape=(image_feature_dim,), name="image_features")
    text_input = Input(shape=(text_feature_dim,), name="text_features")

    # Fusion of features
    x = layers.Concatenate()([image_input, text_input])
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name="fusion_output")(x)

    model = Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def fuse_predictions(image_features, text_features):
    """
    Generate a fused prediction using the pretrained fusion model.
    For now, uses an untrained fusion model as placeholder.
    """
    # Build the fusion model
    fusion_model = build_fusion_model()

    # Convert features to numpy
    image_features = np.array(image_features)
    text_features = np.array(text_features)

    # Predict using the fusion model
    preds = fusion_model.predict([image_features, text_features])

    # Get class label
    class_idx = np.argmax(preds, axis=1)[0]
    class_names = ["Healthy", "Leaf Blight", "Rust", "Nutrient Deficiency"]
    predicted_label = class_names[class_idx] if class_idx < len(class_names) else "Unknown"

    return predicted_label, preds.tolist()
