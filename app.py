from rag_core import setup_qa_chain
import streamlit as st
import os

DB_FAISS_PATH = 'vectorstore/db_faiss'

st.set_page_config(page_title="Financial Report Analysis 📈")
st.title("Financial Report Analysis 📈")
st.info("Ask any question about the company's financial report.")

if not os.path.exists(DB_FAISS_PATH):
    st.error("Vector database not found. Please run `rag_core.py` first to create it.")
    st.stop()

@st.cache_resource
def load_qa_chain():
    return setup_qa_chain()

qa_chain = load_qa_chain()

user_query = st.text_input("Your question:", placeholder="e.g., What were the total revenues?")

if user_query:
    with st.spinner("Thinking..."):
        try:
            answer = qa_chain.invoke(user_query)
            st.success("Here's the answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")