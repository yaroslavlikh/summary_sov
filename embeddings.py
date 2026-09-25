import threading

EMBEDDING_DIM = 384

_model = None
# Without this lock every thread that finds _model unset loads its own copy of the
# model -- seen with parallel eval workers, and reachable in production too, where
# the webhook thread and the background pool both embed.
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    return _model


def embed(text):
    return _get_model().encode(text).tolist()


def embed_batch(texts, batch_size=128):
    return _get_model().encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()


def warm_up():
    _get_model()


def to_vector_literal(embedding):
    return '[' + ','.join(str(x) for x in embedding) + ']'
