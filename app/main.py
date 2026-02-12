from fastapi import FastAPI
from app.llm_model import load_llm
from app.rag_pipeline import build_vector_store, generate_answer

app = FastAPI()

llm = load_llm()

# Build vector store once at startup
build_vector_store("data/company_docs.pdf")


@app.get("/ask")
def ask(question: str):
    answer = generate_answer(llm, question)
    return {"response": answer}
