# srcimage_model.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# List of classes (you can customize later)
CLASS_NAMES = ['Healthy', 'Leaf_Blight', 'Leaf_Spot', 'Rust']

def load_image_model()
    Loads a pre-trained MobileNetV2 model (transfer learning ready).
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    return model

def preprocess_image(img_path)
    Preprocess the image for the model.
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_disease(model, img_path)
    Predicts the disease from the image.
    img_tensor = preprocess_image(img_path)
    preds = model.predict(img_tensor)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)
    return pred_class, float(confidence)

# Example test
if __name__ == __main__
    model = load_image_model()
    result, conf = predict_disease(model, sample_leaf.jpg)
    print(fPredicted {result}, Confidence {conf.2f})
