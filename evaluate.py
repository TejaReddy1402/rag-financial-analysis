# evaluate.py
import pandas as pd
import mlflow
from rag_core import setup_qa_chain
from sentence_transformers.util import cos_sim
# Connects this script to the running MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
# --- 1. Setup ---
eval_df = pd.read_csv("eval_dataset.csv")
qa_chain = setup_qa_chain()

# --- 2. Evaluation Logic ---
def evaluate_model(df):
    results = []
    for index, row in df.iterrows():
        question = row['question']
        ground_truth = row['ground_truth_answer']
        
        # --- UPDATED LINE ---
        # The chain now takes a string directly, not a dictionary
        generated_answer = qa_chain.invoke(question)
        
        # Calculate cosine similarity as a relevance metric
        # Note: The chain from rag_core.py doesn't expose the embedding model directly anymore.
        # This part of the evaluation would need a more complex setup.
        # For now, we will log the answers and can manually review them.
        # We will set similarity_score to a placeholder value like -1.
        similarity_score = -1 
        
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "similarity_score": similarity_score
        })
    return pd.DataFrame(results)

# --- 3. MLflow Logging ---
def run_evaluation():
    print("Starting evaluation...")
    with mlflow.start_run(run_name="RAG Financial Analysis Evaluation") as run:
        evaluation_results_df = evaluate_model(eval_df)
        
        avg_similarity = evaluation_results_df['similarity_score'].mean()
        mlflow.log_metric("average_similarity_score", avg_similarity)
        
        print(f"Average Similarity Score: {avg_similarity}")
        
        results_path = "evaluation_results.csv"
        evaluation_results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, "evaluation-results")
        
        print("Evaluation complete. Results logged to MLflow.")

if __name__ == "__main__":
    run_evaluation()