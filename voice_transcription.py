import io

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
    return _model


def warm_up():
    _get_model()


def transcribe(audio_bytes):
    segments, _ = _get_model().transcribe(io.BytesIO(audio_bytes), language="ru")
    return " ".join(segment.text.strip() for segment in segments).strip()
