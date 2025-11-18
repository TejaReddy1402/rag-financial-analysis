import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

load_dotenv()

DATA_PATH = 'data/financial_report.pdf'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    print("Creating vector database...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
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
    # Load the DB; if it fails, it might not exist yet
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise FileNotFoundError(f"Could not load vector DB at {DB_FAISS_PATH}. Did you run create_vector_db()?") from e
        
    retriever = db.as_retriever()

    class GoogleLLM:
        def __init__(self, model_name="gemini-2.5-flash"):
            self.model = model_name
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
            
        def generate(self, prompt: str) -> str:
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            return response.text

    class SimpleRAGChain:
        def __init__(self, retriever, llm):
            self.retriever = retriever
            self.llm = llm
            self.template = (
                "You are an expert financial analyst. Use the following pieces of context to answer the question.\n"
                "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
            )

        def invoke(self, question: str) -> str:
            docs = self.retriever.invoke(question)
            context = "\n".join([d.page_content for d in docs[:3]])
            prompt = self.template.format(context=context, question=question)
            return self.llm.generate(prompt)

    google_model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    llm = GoogleLLM(model_name=google_model)
    return SimpleRAGChain(retriever=retriever, llm=llm)

if __name__ == "__main__":
    create_vector_db()