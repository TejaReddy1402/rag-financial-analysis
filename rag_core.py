import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential
from google import genai
from google.genai.types import HttpOptions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (API Keys)
load_dotenv()

DATA_DIR = 'data/profiles/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    """Processes PDFs into high-context chunks and saves to FAISS index."""
    if not os.path.exists(DATA_DIR): 
        os.makedirs(DATA_DIR)
        
    all_docs = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not files: 
        return False

    for filename in files:
        loader = PyPDFLoader(os.path.join(DATA_DIR, filename))
        docs = loader.load()
        # Metadata helps the LLM cite specific companies
        company_name = filename.replace(".pdf", "").replace("_", " ")
        for d in docs:
            d.metadata["company"] = company_name
            d.metadata["source"] = filename
        all_docs.extend(docs)

    # OPTIMIZED: Large chunks capture the 'spirit' of CEO commentary and detailed data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    texts = text_splitter.split_documents(all_docs)
    
    # Using local embeddings for speed and cost-efficiency
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    return True

class GoogleLLM:
    """Wrapper for the Gemini 2.5 Flash Model."""
    def __init__(self, model_name="gemini-2.5-flash"):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=HttpOptions(api_version="v1")
        )
        self.model = model_name
        
    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=60))
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, 
            contents=prompt
        )
        return response.text

class SimpleRAGChain:
    """The Logic Chain connecting Retrieval to Generation."""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        # BRANDING UPDATE: Identity in system prompt
        self.template = (
            "SYSTEM: You are the Financial Document Analysing Assistant. Provide a detailed answer using the CONTEXT provided. "
            "If the answer is a vision or strategic point from a CEO, synthesize it fully with professional terminology. "
            "If the information is not present, state that you cannot find it in the current profiles.\n\n"
            "CONTEXT:\n{context}\n\nUSER QUESTION: {question}\n\nFINAL RESPONSE:"
        )

    def invoke(self, question: str):
        # Retrieve the most relevant 8 chunks
        docs = self.retriever.invoke(question)
        context = "\n---\n".join([f"[{d.metadata.get('company')}] {d.page_content}" for d in docs])
        
        # Generate the answer using Gemini 2.5
        answer = self.llm.generate(self.template.format(context=context, question=question))
        
        # Return answer and list of unique sources
        sources = list(set([d.metadata.get('company') for d in docs]))
        return {"answer": answer, "sources": sources}

def setup_qa_chain():
    """Initializes the retriever and the RAG chain for the app."""
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # Load the local vector database
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    # k=8 provides a deep context window for Gemini
    return SimpleRAGChain(db.as_retriever(search_kwargs={"k": 8}), GoogleLLM())