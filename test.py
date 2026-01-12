import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
# Note: Ensure these matches the embedding model used in your rag_core.py
from langchain_huggingface import HuggingFaceEmbeddings 

# 1. Load Environment Variables (for API keys if you use OpenAI, etc.)
load_dotenv()

def run_test_query(query_text):
    print(f"\n--- Testing Query: '{query_text}' ---")
    
    # 2. Define the path to your existing vector store
    DB_FAISS_PATH = os.path.join('vectorstore', 'db_faiss')
    
    # 3. Initialize the same embedding model used during ingestion
    # If you used a different model, update the model_name here.
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    try:
        # 4. Load the local FAISS index
        # 'allow_dangerous_deserialization' is required to load .pkl files locally
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # 5. Perform a Similarity Search
        # This checks if the vector store actually finds relevant text chunks
        docs = db.similarity_search(query_text, k=3)
        
        if not docs:
            print("❌ No relevant documents found. Check if your vectorstore is empty.")
            return

        print(f"✅ Found {len(docs)} relevant chunks:")
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            print(f"\n[Result {i+1}] Source: {source}")
            print(f"Snippet: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Error loading or querying vectorstore: {e}")

if __name__ == "__main__":
    # Test 1: Generic financial query
    run_test_query("What are the key financial highlights for NVIDIA?")
    
    # Test 2: Specific bank query
    run_test_query("Give me a summary of Barclays' performance.")