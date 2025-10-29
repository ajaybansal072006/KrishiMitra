import streamlit as st
import tempfile
import os

from src.image_model import predict_disease
from src.audio_to_text import convert_audio_to_text
from src.text_model import analyze_text
from src.cure_recommender import recommend_cure
from src.fusion_model import fuse_predictions

# -------------------------------------------------------
# 🌾 Streamlit Page Config
# -------------------------------------------------------
st.set_page_config(page_title="KrishiMitra 🌾", layout="centered")

st.title("🌾 KrishiMitra – AI-Powered Crop Disease Assistant")
st.write("Upload a leaf image and describe the issue (voice or text). The system analyzes both inputs to suggest a cure.")

# -------------------------------------------------------
# 🖼️ Step 1: Upload Crop Image
# -------------------------------------------------------
st.header("1️⃣ Upload Crop Image")
uploaded_image = st.file_uploader("Upload an image of the affected leaf", type=["jpg", "jpeg", "png"])

# -------------------------------------------------------
# 🎙️ Step 2: Audio/Text Input
# -------------------------------------------------------
st.header("2️⃣ Describe the Issue")
option = st.radio("Choose input type:", ["🎙️ Record / Upload Audio", "⌨️ Type Text"])

text_summary = ""

if option == "🎙️ Record / Upload Audio":
    audio_file = st.file_uploader("Upload audio (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        with st.spinner("🎧 Converting speech to text using Whisper..."):
            try:
                text_summary = convert_audio_to_text(tmp_path)
                st.success("Audio converted successfully!")
                st.markdown(f"🗣️ **Transcribed Text:** *{text_summary}*")
            except Exception as e:
                st.error(f"❌ Whisper Error: {e}")
        os.remove(tmp_path)
else:
    text_summary = st.text_area("Describe what you see on the plant", placeholder="E.g., yellow spots, dried edges, fungus...")

# -------------------------------------------------------
# 🔍 Step 3: Analyze & Recommend
# -------------------------------------------------------
if st.button("🔍 Analyze & Recommend"):
    if not uploaded_image:
        st.warning("⚠️ Please upload a leaf image!")
    elif not text_summary.strip():
        st.warning("⚠️ Please provide a description or audio input!")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_image.read())
            image_path = tmp_img.name

        # --- Image Model ---
        with st.spinner("🔬 Analyzing image..."):
            img_label, img_conf, image_features = predict_disease(image_path)
            st.markdown(f"🌿 **Image Model Prediction:** `{img_label}` ({img_conf:.2f} confidence)")

        # --- Text Model ---
        with st.spinner("💬 Analyzing text description..."):
            text_label, text_features = analyze_text(text_summary)
            st.markdown(f"🧠 **Text Model Prediction:** `{text_label}` (features extracted ✅)")

        # --- Fusion Model ---
        with st.spinner("⚡ Combining both analyses..."):
            final_label, fusion_conf, _ = fuse_predictions(image_features, text_features)
            st.markdown(f"## ✅ Final Diagnosis: `{final_label}` ({fusion_conf:.2f} confidence)")

        # --- Cure Recommendation ---
        cure = recommend_cure(final_label)
        if isinstance(cure, list):
            cure = cure[0] if len(cure) > 0 else "No specific recommendation available."
        st.success(f"💊 **Suggested Cure:** {cure}")

        os.remove(image_path)
