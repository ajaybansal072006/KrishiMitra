import streamlit as st
import tempfile
import os
from src.image_model import predict_disease
from src.audio_to_text import convert_audio_to_text
from src.text_model import analyze_text
from src.cure_recommender import recommend_cure
from src.fusion_model import fuse_predictions

st.set_page_config(page_title="KrishiMitra 🌾", layout="centered")

st.title("🌾 KrishiMitra – AI-Powered Crop Disease Assistant")
st.write("Upload a leaf image and describe the problem (via voice or text). The system will analyze both to recommend a cure.")

# --- Upload image ---
st.header("1️⃣ Upload Crop Image")
uploaded_image = st.file_uploader("Upload an image of the affected leaf", type=["jpg", "jpeg", "png"])

# --- Upload or type audio/text ---
st.header("2️⃣ Describe the Issue")
option = st.radio("Choose input type:", ["🎙️ Record / Upload Audio", "⌨️ Type Text"])

text_summary = ""

if option == "🎙️ Record / Upload Audio":
    audio_file = st.file_uploader("Upload audio (mp3, wav, m4a)", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        with st.spinner("Converting speech to text using Whisper..."):
            try:
                text_summary = convert_audio_to_text(tmp_path)
                st.success("Audio converted successfully!")
                st.write(f"🗣️ Transcribed text: *{text_summary}*")
            except Exception as e:
                st.error(f"Error in Whisper: {e}")
        os.remove(tmp_path)

else:
    text_summary = st.text_area("Describe what you see on the plant", placeholder="E.g., yellow spots, dried edges, fungus...")

# --- Process both inputs ---
if st.button("🔍 Analyze & Recommend"):
    if not uploaded_image:
        st.warning("Please upload a leaf image!")
    elif not text_summary.strip():
        st.warning("Please provide text or audio description!")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_image.read())
            image_path = tmp_img.name

        with st.spinner("Analyzing image..."):
            img_label, img_conf = predict_disease(image_path)
            st.write(f"🌿 Image Model: **{img_label}** ({img_conf:.2f} confidence)")

        with st.spinner("Analyzing description..."):
            text_result = analyze_text(text_summary)
            st.write(f"🧠 Text Model: **{text_result}**")

        final_pred = fuse_predictions((img_label, img_conf), text_summary)
        st.subheader(f"✅ Final Diagnosis: {final_pred}")

        cure = recommend_cure(final_pred)
        st.success(f"💊 Suggested Cure: {cure}")

        os.remove(image_path)
