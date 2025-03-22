import os
import pandas as pd
import numpy as np
import json
import time
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import score as bert_score

# Import the MedlinePlusVectorizer class
from Vectorizer import MedlinePlusVectorizer  # Adjust the import path as needed

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LOG_FILE = "rag_evaluation.log"
RESULTS_FILE = "rag_evaluation_results.csv"  # Changed to .csv file
SUMMARY_FILE = "rag_evaluation_summary.csv"  # Separate file for summary stats
METADATA_FILE = "rag_evaluation_metadata.csv"  # Separate file for metadata
TEST_CASES_FILE = "cleaned_file_with_context_excel.xlsx"
COLLECTION_NAME = "medlineplus_collection"

# Initialize evaluation models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Setup logger
def setup_logger():
    """Initialize logging file if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,question,generated_answer,reference_answer,cosine_similarity,rouge_score,bert_score_precision,bert_score_recall,bert_score_f1,final_score\n")

def log_interaction(question, generated, reference, metrics):
    """Log interaction with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_metrics(generated, reference):
    """Calculate cosine similarity, ROUGE score, and BERTScore metrics"""
    # Cosine Similarity (using SentenceTransformer)
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

    # ROUGE Score
    rouge_score = rouge.score(reference, generated)['rougeL'].fmeasure

    # BERTScore
    bert_score_precision, bert_score_recall, bert_score_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
    bert_score_precision = bert_score_precision.mean().item()
    bert_score_recall = bert_score_recall.mean().item()
    bert_score_f1 = bert_score_f1.mean().item()

    # Final Score (weighted average)
    final_score = (
        cosine_sim * 0.10 +           # Weight: 10%
        rouge_score * 0.20 +          # Weight: 20%
        bert_score_precision * 0.20 + # Weight: 20%
        bert_score_recall * 0.20 +    # Weight: 20%
        bert_score_f1 * 0.30          # Weight: 30%
    )

    return {
        "cosine_similarity": float(cosine_sim),
        "rouge_score": rouge_score,
        "bert_score_precision": bert_score_precision,
        "bert_score_recall": bert_score_recall,
        "bert_score_f1": bert_score_f1,
        "final_score": final_score
    }

def load_test_cases(filepath):
    """Parses test cases from an Excel file into a DataFrame."""
    df = pd.read_excel(filepath)
    
    # Check if required columns exist
    required_columns = ["Question", "Answer"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the Excel file.")
    
    print(f"✅ Loaded {len(df)} test cases.")
    return df

def evaluate_rag_system():
    """Evaluate the RAG system on the test cases"""
    setup_logger()
    
    try:
        # Load test cases
        df = load_test_cases(TEST_CASES_FILE)
        
        if df.empty:
            print("⚠️ No test cases found. Please check the file format.")
            return
        
        # Process ALL test cases instead of limiting to 20
        print(f"⚠️ Processing all {len(df)} test cases. This might take a while...")
        
        # Initialize results DataFrame
        results_df = pd.DataFrame(columns=[
            "question", "generated_answer", "reference_answer",
            "cosine_similarity", "rouge_score", "bert_score_precision", 
            "bert_score_recall", "bert_score_f1", "final_score"
        ])
        
        # Initialize the MedlinePlusVectorizer instead of MedicalRAG
        print("Initializing RAG system using MedlinePlusVectorizer...")
        rag_system = MedlinePlusVectorizer(collection_name=COLLECTION_NAME)
        
        # Process test cases
        pbar = tqdm(total=len(df), desc="Evaluating RAG")
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Get the question and reference answer
                question = row["Question"]
                reference = row["Answer"]
                
                # Query the RAG system using the vectorizer's query_with_rag method
                generated, _ = rag_system.query_with_rag(question)
                
                # Skip processing if an error occurred
                if isinstance(generated, str) and generated.startswith("Error"):
                    print(f"⚠️ Skipping evaluation for question {idx+1} due to error in response generation.")
                    continue
                
                # Calculate metrics
                metrics = calculate_metrics(
                    generated=generated,
                    reference=reference
                )
                
                # Log the interaction
                log_interaction(
                    question=question,
                    generated=generated,
                    reference=reference,
                    metrics=metrics
                )
                
                # Append to results
                results.append({
                    "question": question,
                    "generated_answer": generated,
                    "reference_answer": reference,
                    **metrics
                })
                
                pbar.update(1)
                pbar.set_postfix({"Processed": f"{idx+1}/{len(df)}", "Score": f"{metrics['final_score']:.2f}"})
                
                # Small delay to avoid hitting rate limits too quickly
                time.sleep(0.5)
                
            except Exception as e:
                error_msg = f"Error processing case {idx}: {str(e)}"
                print(f"❌ {error_msg}")
                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps({
                        "timestamp": datetime.now().isoformat(),
                        "error": error_msg,
                        "question": row.get("Question", "")
                    }) + "\n")
                continue
        
        pbar.close()
        
        # Save results to CSV files
        if results:
            results_df = pd.DataFrame(results)
            
            # Save detailed results to CSV
            results_df.to_csv(RESULTS_FILE, index=False, encoding='utf-8')
            
            # Create and save summary statistics
            metrics_columns = ['cosine_similarity', 'rouge_score', 'bert_score_precision', 
                              'bert_score_recall', 'bert_score_f1', 'final_score']
            
            summary_data = {
                'Metric': metrics_columns,
                'Average': [results_df[col].mean() for col in metrics_columns],
                'Min': [results_df[col].min() for col in metrics_columns],
                'Max': [results_df[col].max() for col in metrics_columns],
                'StdDev': [results_df[col].std() for col in metrics_columns]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(SUMMARY_FILE, index=False, encoding='utf-8')
            
            # Create and save metadata
            metadata = pd.DataFrame({
                'Metadata': ['Evaluation Date', 'Number of Test Cases', 'Collection Name'],
                'Value': [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                         len(results_df), 
                         COLLECTION_NAME]
            })
            metadata.to_csv(METADATA_FILE, index=False, encoding='utf-8')
            
            print(f"\n✅ Evaluation complete! Results saved to:")
            print(f"  - Detailed results: {RESULTS_FILE}")
            print(f"  - Summary statistics: {SUMMARY_FILE}")
            print(f"  - Metadata: {METADATA_FILE}")
            print(f"  - Logs: {LOG_FILE}")
            
            print("\nFinal Metrics Summary:")
            for metric, value in zip(metrics_columns, summary_data['Average']):
                print(f"  {metric}: {value:.4f}")
        else:
            print("\n⚠️ No results were successfully processed.")
    
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")

if __name__ == "__main__":
    evaluate_rag_system()