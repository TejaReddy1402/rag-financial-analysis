RAG Financial Document Analysis 📈
This project is a sophisticated Retrieval-Augmented Generation (RAG) application that allows users to ask questions about dense financial documents (like a 10-K annual report) and receive accurate, context-aware answers.

The application is built with a modern Python stack and integrates a powerful API-based Large Language Model (Google Gemini). It also includes a complete MLOps evaluation pipeline using MLflow and a CI/CD workflow with GitHub Actions for automated testing.

🚀 Live Demo - Financial Report Analysis · Streamlit
You can access the live application here: https://www.google.com/search?q=https://rag-financial-analysis-zxxqyetkv7ecuvgzz8jffy.streamlit.app/

✨ Key Features
Interactive Chat Interface: A user-friendly web app built with Streamlit.

PDF Document Ingestion: Processes and understands complex PDF documents.

Accurate Q&A: Implements a RAG pipeline using a FAISS vector store for efficient semantic search.

High-Quality Generation: Integrates the Google Gemini 1.5 Flash model via API for state-of-the-art answers.

Automated Evaluation: Uses an MLflow pipeline to test the RAG system's quality against a ground-truth dataset.

CI/CD Workflow: A GitHub Actions pipeline automatically runs the evaluation script on every push to the main branch.

🛠️ Tech Stack
Programming Language: Python

Core Libraries: LangChain, Streamlit

Vector Database: FAISS (for local semantic search)

LLM: Google Gemini API (via langchain-google-genai)

Embeddings: Hugging Face sentence-transformers

MLOps: MLflow

CI/CD: GitHub Actions

Environment: Conda

📂 Project Structure
/rag-financial-analysis
|
|-- .github/workflows/ci-cd.yml # GitHub Actions workflow
|-- data/                       # Input PDFs
|-- vectorstore/                # FAISS index
|-- .env                        # Secret API key (ignored)
|-- .gitignore                  # Files for git to ignore
|-- app.py                      # Streamlit web application
|-- evaluate.py                 # MLflow evaluation script
|-- eval_dataset.csv            # Ground-truth Q&A data
|-- rag_core.py                 # Core RAG pipeline logic
|-- requirements.txt            # Project dependencies
|-- README.md                   # This file

⚙️ Setup and Local Installation
To run this project on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/TejaReddy1402/rag-financial-analysis.git](https://github.com/TejaReddy1402/rag-financial-analysis.git)
cd rag-financial-analysis

Set up the Conda environment:

# (Optional but recommended)
conda create --name rag_env python=3.10
conda activate rag_env

Install dependencies:

python -m pip install -r requirements.txt

Set up your API Key:

Create a .env file in the root directory.

Add your Google AI API key to it: GOOGLE_API_KEY="YOUR_API_KEY_HERE"

Add Data:

Place your financial report PDF inside the data/ folder and name it financial_report.pdf.

Build the Vector Database:

Run the core script once to process the PDF and create the vector store.

python rag_core.py

Launch the App:

streamlit run app.py

Journey & Development Process
This project evolved significantly through a series of challenges and solutions, demonstrating a practical approach to building modern AI applications.

Initial RAG Build: The project started with a standard RAG pipeline using local, open-source models from Hugging Face (Flan-T5, Mistral, TinyLlama).

Local Model Challenges:

Generation Quality: Early models often failed to summarize, instead dumping raw, unformatted context.

Network Instability: Larger, more powerful models like Mistral-7B (~15 GB) repeatedly failed to download due to unstable network connections (IncompleteRead errors).

Capability vs. Size: The smaller TinyLlama model downloaded successfully but was not powerful enough to reason over the complex financial tables, producing garbled answers.

Pivot to an API-based Solution:

To overcome the limitations of local models, the project was refactored to use an API. This eliminated download issues and provided access to a state-of-the-art model.

The Google Gemini API (gemini-1.5-flash) was chosen for its excellent performance and generous free tier. A custom prompt template was engineered to instruct the model to act as a financial analyst.

Environment & Security:

Resolved fundamental environment configuration issues by reinstalling Anaconda and using explicit python -m pip commands to ensure dependencies were installed correctly.

Fixed a critical security vulnerability by removing a leaked API key from Git history and implementing the use of .gitignore and Streamlit's Secrets manager.

MLOps and Automation:

An evaluation script (evaluate.py) was developed to test the RAG chain against a set of ground-truth questions and answers.

MLflow was integrated to log experiment runs and artifacts, allowing for tracking of the system's quality over time.

A GitHub Actions workflow was created to establish a CI/CD pipeline, automatically running the evaluation script on every push to the main branch, ensuring that new changes do not break the application's core functionality.