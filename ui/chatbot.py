import streamlit as st
import requests

st.title("GenAI Q&A Chatbot")
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if question:
        response = requests.post("http://127.0.0.1:3750/ask", json={"question": question}).json()
        st.write("Answer:", response["answer"])
    else:
        st.write("Please enter a question.")
