# src/text_audio_model.py
import speech_recognition as sr
from transformers import pipeline

def audio_to_text(audio_file):
    """Converts a .wav audio file to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language='en-IN')
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Speech recognition service error"

def analyze_text_symptoms(text):
    """Analyzes text description to summarize or interpret symptoms."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=25, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Example test
if __name__ == "__main__":
    text = audio_to_text("sample_audio.wav")
    print("Audio to text:", text)
    print("Text summary:", analyze_text_symptoms("Leaves have yellow spots and dry edges."))
