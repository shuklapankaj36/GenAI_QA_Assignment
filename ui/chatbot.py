import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from model.run_inference import answer_question

st.title("GenAI Q&A Chatbot")
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if question:
        response = answer_question(question)
        if isinstance(response, dict):
            st.write("### Answer:")
            st.write(response['answer'])
            st.write("### Context:")
            st.write(response['context'])
        else:
            st.error(f"Response: {response}")
    else:
        st.warning("Please enter a question.")
