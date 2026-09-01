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


def to_vector_literal(embedding):
    return '[' + ','.join(str(x) for x in embedding) + ']'
