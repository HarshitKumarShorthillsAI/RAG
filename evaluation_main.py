import os
import pandas as pd
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from tqdm import tqdm
from dotenv import load_dotenv
from bert_score import score as bert_score

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
LOG_FILE = "qa_interactions.log"
RESULTS_FILE = "evaluation_results_groundtruth.xlsx"
TEST_CASES_FILE = "cleaned_file_with_context_excel.xlsx"
VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

# Metric weights for final score calculation
METRIC_WEIGHTS = {
    "rouge_score": 0.25,
    "cosine_similarity": 0.25,
    "bert_score_f1": 0.5  # Using F1 as the main BERT Score metric
}

if MISTRAL_MODEL not in VALID_MODELS:
    raise ValueError(f"Invalid model '{MISTRAL_MODEL}'. Choose from {VALID_MODELS}")

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Setup logger
def setup_logger():
    """Initialize logging file if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,question,context,generated_answer,reference_answer,rouge_score,cosine_similarity,bert_score_precision,bert_score_recall,bert_score_f1,final_score\n")

def log_interaction(question, context, generated, reference, metrics):
    """Log interaction with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "context": context,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_metrics(generated, reference):
    """Calculate metrics and combined final score"""
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
    
    # Calculate final score (weighted average)
    final_score = (
        METRIC_WEIGHTS["rouge_score"] * rouge_score +
        METRIC_WEIGHTS["cosine_similarity"] * float(cosine_sim) +
        METRIC_WEIGHTS["bert_score_f1"] * bert_score_f1
    )

    return {
        "rouge_score": rouge_score,
        "cosine_similarity": float(cosine_sim),
        "bert_score_precision": bert_score_precision,
        "bert_score_recall": bert_score_recall,
        "bert_score_f1": bert_score_f1,
        "final_score": final_score
    }

def load_test_cases(filepath):
    """Parses test cases from an Excel file into a DataFrame."""
    try:
        # First try to load with default engine
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"Trying alternative Excel engine due to: {str(e)}")
        # Try with openpyxl engine
        df = pd.read_excel(filepath, engine="openpyxl")
    
    print(f"✅ Loaded {len(df)} test cases. Processing all questions.")
    return df

def qa_pipeline(question, context):
    """Query Mistral API with context, with exponential backoff retry handling."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Modified to include context in the prompt
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping with question answering. Use the provided context to answer the question accurately."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}
        ],
        "max_tokens": 150,
        "temperature": 0.7
    }

    max_retries = 5
    backoff_time = 2  # Initial wait time in seconds

    for attempt in range(max_retries):
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        
        elif response.status_code == 429:  # Rate limit error
            wait_time = backoff_time * (2 ** attempt)  # Exponential backoff
            print(f"⚠️ Rate limit hit. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)
        
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return "Error generating response"
    
    print("🚨 Max retries reached. Skipping question.")
    return "Error generating response"

def process_test_cases():
    """Main processing function with logging and retry handling"""
    setup_logger()
    
    df = load_test_cases(TEST_CASES_FILE)
    
    if df.empty:
        print("⚠️ No test cases found. Please check the file format.")
        return

    # Verify column names
    required_columns = ["Question", "Answer", "Context"]
    for col in required_columns:
        if col not in df.columns:
            print(f"❌ Required column '{col}' not found in Excel file. Available columns: {list(df.columns)}")
            return

    # Initialize results collection
    results = []
    
    pbar = tqdm(total=len(df), desc="Processing test cases")

    for idx, row in df.iterrows():
        try:
            # Generate answer WITH context
            generated = qa_pipeline(row["Question"], row["Context"])
            
            metrics = calculate_metrics(
                generated=generated,
                reference=row["Answer"]
            )

            log_interaction(
                question=row["Question"],
                context=row["Context"],
                generated=generated,
                reference=row["Answer"],
                metrics=metrics
            )

            # Add to results list
            results.append({
                "question": row["Question"],
                "context": row["Context"],
                "generated_answer": generated,
                "reference_answer": row["Answer"],
                **metrics
            })

            pbar.update(1)
            metrics_summary = f"ROUGE: {metrics['rouge_score']:.2f}, Cosine: {metrics['cosine_similarity']:.2f}, BERT-F1: {metrics['bert_score_f1']:.2f}, Final: {metrics['final_score']:.2f}"
            pbar.set_postfix({"Metrics": metrics_summary})
            
        except Exception as e:
            print(f"Error processing case {idx}: {str(e)}")
            error_msg = f"Error processing case {idx}: {str(e)}"
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "case_idx": idx
                }) + "\n")
            pbar.update(1)
            continue

    pbar.close()
    
    # Create DataFrame from results and save to Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(RESULTS_FILE, index=False, engine='openpyxl')
    
    print(f"\n✅ Processing complete! Processed {len(results)} questions out of {len(df)} total. Results saved to {RESULTS_FILE}. Logs in {LOG_FILE}.")
    
    return results_df

def calculate_grade(score):
    """Convert numerical score to letter grade"""
    if score >= 0.90:
        return "A (Excellent)"
    elif score >= 0.80:
        return "B (Good)"
    elif score >= 0.70:
        return "C (Average)"
    elif score >= 0.60:
        return "D (Below Average)"
    else:
        return "F (Poor)"

if __name__ == "__main__":
    print("\n📊 Evaluation weights:")
    for metric, weight in METRIC_WEIGHTS.items():
        print(f"  - {metric}: {weight*100}%")
    print()
    
    results_df = process_test_cases()
    
    try:
        print("\nFinal Metrics Summary:")
        metrics_cols = ['rouge_score', 'cosine_similarity', 'bert_score_precision', 
                        'bert_score_recall', 'bert_score_f1', 'final_score']
        summary = results_df[metrics_cols].mean()
        
        # Format for readability
        print("=" * 50)
        print(f"ROUGE Score:          {summary['rouge_score']:.4f}")
        print(f"Cosine Similarity:    {summary['cosine_similarity']:.4f}")
        print(f"BERT Score Precision: {summary['bert_score_precision']:.4f}")
        print(f"BERT Score Recall:    {summary['bert_score_recall']:.4f}")
        print(f"BERT Score F1:        {summary['bert_score_f1']:.4f}")
        print("-" * 50)
        print(f"FINAL SCORE:          {summary['final_score']:.4f}  {calculate_grade(summary['final_score'])}")
        print("=" * 50)
        
        # Also save summary to a separate sheet in the Excel file
        with pd.ExcelWriter(RESULTS_FILE, engine='openpyxl', mode='a') as writer:
            summary_df = pd.DataFrame([summary], index=['Average'])
            summary_df.loc['Grade'] = ['', '', '', '', '', calculate_grade(summary['final_score'])]
            summary_df.to_excel(writer, sheet_name='Summary')
            
        print(f"Summary metrics also added to '{RESULTS_FILE}' in the 'Summary' sheet.")
    except Exception as e:
        print(f"Error generating summary metrics: {str(e)}")