# GenAI RAG Chatbot 🚀

A full-stack Retrieval-Augmented Generation (RAG) chatbot built using:

- FastAPI
- Streamlit
- HuggingFace Transformers
- SentenceTransformers
- FAISS

## Features

- PDF document ingestion
- Vector embeddings
- Semantic search using FAISS
- LLM response generation
- Streamlit Chat UI

## How to Run

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Start Backend

uvicorn app.main:app --reload

### 3. Start Frontend

streamlit run streamlit_app.py
