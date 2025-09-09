import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATA_PATH = 'data/financial_report.pdf'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    print("Creating vector database...")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print("Vector database created successfully.")

def setup_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    template = """
    You are an expert financial analyst. Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know. Do not make up an answer.
    Provide a concise, professional answer.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:
    """
    
    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        create_vector_db()

    qa_chain = setup_qa_chain()
    query = "What were the total revenues in the last fiscal year?"
    print(f"\nQuery: {query}")
    result = qa_chain.invoke(query) 
    print(f"Answer: {result}")