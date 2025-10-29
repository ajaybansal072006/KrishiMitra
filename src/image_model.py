# src/image_model.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------------------------------------------
# 🧠 Build the model (EfficientNetB0 backbone)
# -------------------------------------------------------
def build_image_model(num_classes=4, feature_dim=256):
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(feature_dim, activation='relu', name="image_features")(x)
    output = layers.Dense(num_classes, activation='softmax', name="image_output")(x)

    model = Model(inputs=base_model.input, outputs=[output, x])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------------------------------------
# 📸 Load model and predict disease
# -------------------------------------------------------
# Load your trained weights if available
# For now, use imagenet weights directly
image_model = build_image_model(num_classes=4, feature_dim=256)

def predict_disease(img_path):
    """
    Takes an image path and returns (predicted_label, confidence_score)
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds, _ = image_model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(np.max(preds[0]))

    # ⚠️ Replace with your actual crop disease labels
    class_labels = ["Leaf_Blight", "Rust", "Healthy", "Bacterial_Spot"]
    predicted_label = class_labels[class_idx]

    return predicted_label, confidence
