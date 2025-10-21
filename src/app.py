# src/app.py
import streamlit as st
from image_model import load_image_model, predict_disease
from text_audio_model import audio_to_text, analyze_text_symptoms
from fusion_model import fuse_predictions
from gtts import gTTS
import os

st.set_page_config(page_title="KrishiMitra", page_icon="🌾", layout="centered")

st.title("🌾 KrishiMitra - Multimodal Crop Doctor")
st.markdown("#### Upload a leaf image and optionally record or upload an audio description.")

# Upload inputs
uploaded_image = st.file_uploader("📸 Upload a Leaf Image", type=["jpg", "jpeg", "png"])
uploaded_audio = st.file_uploader("🎙️ Upload an Audio File (Optional)", type=["wav"])

# Load model only once
@st.cache_resource
def get_model():
    return load_image_model()

model = get_model()

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("🔍 Analyzing image...")
    prediction = predict_disease(model, uploaded_image)
    st.success(f"🩺 Image Model Prediction: **{prediction[0]}** ({prediction[1]*100:.1f}% confidence)")

text_summary = None
if uploaded_audio:
    st.write("🎧 Processing audio...")
    text = audio_to_text(uploaded_audio)
    st.write("🗣️ Recognized Text:", text)
    text_summary = analyze_text_symptoms(text)
    st.write("🧠 Summarized Symptoms:", text_summary)

# Combine predictions if both are available
if uploaded_image and text_summary:
    final_pred = fuse_predictions(prediction, text_summary)
    st.success(f"🌿 Final Fused Prediction: **{final_pred}**")

    # Voice output
    tts = gTTS(f"The detected disease is {final_pred}. Please take necessary precautions.", lang='en')
    tts.save("result.mp3")
    audio_file = open("result.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

st.markdown("---")
st.caption("Developed by Team KrishiMitra | Hackathon 2025")
