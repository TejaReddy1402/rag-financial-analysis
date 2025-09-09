# RAG Financial Document Analysis 📈

This project is a sophisticated Retrieval-Augmented Generation (RAG) application that allows users to ask questions about dense financial documents (like a 10-K annual report) and receive accurate, context-aware answers.

The application is built with a modern Python stack and integrates a powerful API-based Large Language Model (Google Gemini). It also includes a complete MLOps evaluation pipeline using MLflow and a CI/CD workflow with GitHub Actions for automated testing.

## 🚀 Live Demo - Financial Report Analysis · Streamlit

**You can access the live application here:** https://rag-financial-analysis-zxxqyetkv7ecuvgzz8jffy.streamlit.app/

## ✨ Key Features

* **Interactive Chat Interface:** A user-friendly web app built with Streamlit.
* **PDF Document Ingestion:** Processes and understands complex PDF documents.
* **Accurate Q&A:** Implements a RAG pipeline using a FAISS vector store for efficient semantic search.
* **High-Quality Generation:** Integrates the Google Gemini 1.5 Flash model via API for state-of-the-art answers.
* **Automated Evaluation:** Uses an MLflow pipeline to test the RAG system's quality against a ground-truth dataset.
* **CI/CD Workflow:** A GitHub Actions pipeline automatically runs the evaluation script on every push to the main branch.

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Core Libraries:** LangChain, Streamlit
* **Vector Database:** FAISS (for local semantic search)
* **LLM:** Google Gemini API (via `langchain-google-genai`)
* **Embeddings:** Hugging Face `sentence-transformers`
* **MLOps:** MLflow
* **CI/CD:** GitHub Actions
* **Environment:** Conda

## 📂 Project Structure
