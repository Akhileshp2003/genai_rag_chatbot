from sentence_transformers import SentenceTransformer


def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(model, texts):
    return model.encode(texts)
