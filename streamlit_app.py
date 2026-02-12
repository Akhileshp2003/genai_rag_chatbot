import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="GenAI RAG Chatbot", page_icon="🤖")

st.title("🤖 GenAI RAG Chatbot")
st.write("Ask questions from your uploaded PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI
    response = requests.get(API_URL, params={"question": prompt})

    if response.status_code == 200:
        answer = response.json()["response"]
    else:
        answer = "Error connecting to backend."

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)
