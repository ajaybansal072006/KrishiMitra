import whisper

def convert_audio_to_text(audio_path: str):
    """
    Transcribes user’s voice describing plant symptoms into text.
    """
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    return result['text']
