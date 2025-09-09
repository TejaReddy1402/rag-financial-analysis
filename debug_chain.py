from rag_core import setup_qa_chain
import pprint # Use pprint for a cleaner dictionary print

print("Setting up the RAG chain for debugging...")
qa_chain = setup_qa_chain()

# Use the same query that failed in the app
query = "What were the total revenues in the last fiscal year?"
print(f"Invoking chain with query: '{query}'")

# Get the full result
result = qa_chain.invoke({"query": query})

print("\n--- RAW RESULT ---")
pprint.pprint(result)
print("--- END OF RAW RESULT ---")