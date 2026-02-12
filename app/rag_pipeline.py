import faiss
import numpy as np
import os
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from app.embeddings import load_embedding_model, embed_texts


VECTOR_PATH = "vector_store/faiss_index.bin"
DOC_PATH = "vector_store/documents.npy"


def build_vector_store(pdf_path):
    text = load_pdf(pdf_path)
    chunks = split_text(text)

    model = load_embedding_model()
    embeddings = embed_texts(model, chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, VECTOR_PATH)
    np.save(DOC_PATH, np.array(chunks))


def retrieve(query, k=3):
    model = load_embedding_model()
    query_embedding = model.encode([query])

    index = faiss.read_index(VECTOR_PATH)
    documents = np.load(DOC_PATH, allow_pickle=True)

    distances, indices = index.search(np.array(query_embedding), k)

    results = [documents[i] for i in indices[0]]
    return results


def generate_answer(llm, question):
    contexts = retrieve(question)
    context_text = "\n".join(contexts)

    prompt = f"""
    Answer using only the context below.

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

    response = llm(prompt)
    return response[0]["generated_text"]
