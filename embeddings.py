EMBEDDING_DIM = 384

_model = None


def _get_model():
    global _model
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
