import streamlit as st
import os
from rag_core import setup_qa_chain

st.set_page_config(page_title="Financial Report Analysis")
st.title("Financial Report Analysis 📈")

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Check if the vector database exists. If not, ask user to wait or run script.
if not os.path.exists(DB_FAISS_PATH):
    st.warning("Vector database not found. It will be created locally if you run rag_core.py, but for cloud deployment, ensure the 'vectorstore' folder is pushed to GitHub.")
    st.stop()

@st.cache_resource
def get_chain():
    return setup_qa_chain()

try:
    chain = get_chain()
except Exception as e:
    st.error(f"Error loading chain: {e}")
    st.stop()

user_query = st.text_input("Your question:", placeholder="e.g., What were the total revenues?")

if user_query:
    with st.spinner("Thinking..."):
        try:
            answer = chain.invoke(user_query)
            st.success("Here's the answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")