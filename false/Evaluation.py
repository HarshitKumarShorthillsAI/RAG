import os
import pandas as pd
import re
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from evaluate import load  # For exact match
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
import torch
from bert_score import score as bert_score  # Import BERTScore
from sklearn.metrics import precision_score  # Import precision_score

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
LOG_FILE = "qa_interactions.log"
RESULTS_FILE = "evaluation_results.csv"
TEST_CASES_FILE = "cleaned_file_with_context.ods"  # Change to ODS file
VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

if MISTRAL_MODEL not in VALID_MODELS:
    raise ValueError(f"Invalid model '{MISTRAL_MODEL}'. Choose from {VALID_MODELS}")

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
exact_match_metric = load("exact_match")

# Initialize BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Setup logger
def setup_logger():
    """Initialize logging file if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,context,question,generated_answer,reference_answer,exact_match,cosine_similarity,rouge_score,bert_similarity,bert_score_precision,bert_score_recall,bert_score_f1,precision_score,final_score\n")

def log_interaction(context, question, generated, reference, metrics):
    """Log interaction with timestamp"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "context": context,
        "question": question,
        "generated_answer": generated,
        "reference_answer": reference,
        **metrics
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def calculate_bert_similarity(text1, text2):
    """Calculate BERT-based semantic similarity between two texts"""
    inputs = bert_tokenizer([text1, text2], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
    return cosine_sim

def calculate_precision(generated, reference):
    """Calculate precision score at the token level"""
    # Tokenize the generated and reference answers
    gen_tokens = set(generated.split())
    ref_tokens = set(reference.split())
    
    # Create binary vectors for precision calculation
    all_tokens = gen_tokens.union(ref_tokens)
    gen_binary = [1 if token in gen_tokens else 0 for token in all_tokens]
    ref_binary = [1 if token in ref_tokens else 0 for token in all_tokens]
    
    # Calculate precision
    if sum(ref_binary) == 0:
        return 0.0  # Avoid division by zero
    precision = precision_score(ref_binary, gen_binary, zero_division=0)
    return precision

def calculate_metrics(generated, reference):
    """Calculate exact match, cosine similarity, ROUGE score, BERT similarity, BERTScore, and precision score"""
    # Exact Match
    exact_match = exact_match_metric.compute(
        predictions=[generated],
        references=[reference]
    )["exact_match"]

    # Cosine Similarity (using SentenceTransformer)
    emb_gen = similarity_model.encode(generated)
    emb_ref = similarity_model.encode(reference)
    cosine_sim = np.dot(emb_gen, emb_ref) / (np.linalg.norm(emb_gen) * np.linalg.norm(emb_ref))

    # ROUGE Score
    rouge_score = rouge.score(reference, generated)['rougeL'].fmeasure

    # BERT Similarity (using custom BERT embeddings)
    bert_sim = calculate_bert_similarity(generated, reference)

    # BERTScore
    bert_score_precision, bert_score_recall, bert_score_f1 = bert_score([generated], [reference], lang="en", model_type="bert-base-uncased")
    bert_score_precision = bert_score_precision.mean().item()
    bert_score_recall = bert_score_recall.mean().item()
    bert_score_f1 = bert_score_f1.mean().item()

    # Precision Score
    precision = calculate_precision(generated, reference)

    # Final Score (weighted average)
    final_score = (
        exact_match * 0.05 +          # Weight: 5%
       cosine_sim * 0.10 +           # Weight: 10%
       rouge_score * 0.20 +          # Weight: 20%
       bert_sim * 0.10 +             # Weight: 10%
       bert_score_precision * 0.15 + # Weight: 15%
       bert_score_recall * 0.15 +    # Weight: 15%
       bert_score_f1 * 0.10 +        # Weight: 10%
       precision * 0.15 
    )

    return {
        "exact_match": exact_match,
        "cosine_similarity": float(cosine_sim),
        "rouge_score": rouge_score,
        "bert_similarity": bert_sim,
        "bert_score_precision": bert_score_precision,
        "bert_score_recall": bert_score_recall,
        "bert_score_f1": bert_score_f1,
        "precision_score": precision,
        "final_score": final_score
    }

def load_test_cases(filepath):
    """Parses test cases from an ODS file into a DataFrame."""
    df = pd.read_excel(filepath, engine="odf")
    print(f"✅ Loaded {len(df)} test cases.")
    return df

def qa_pipeline(question, context=""):
    """Query Mistral API with exponential backoff retry handling."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant helping with RAG-based question answering."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"}
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

    if not os.path.exists(RESULTS_FILE):
        pd.DataFrame(columns=[
            "question", "context", "generated_answer", "reference_answer",
            "exact_match", "cosine_similarity", "rouge_score", "bert_similarity",
            "bert_score_precision", "bert_score_recall", "bert_score_f1", "precision_score", "final_score"
        ]).to_csv(RESULTS_FILE, index=False)

    # Remove or comment out the line that limits the DataFrame to the first 10 rows
    # df = df.head(10)

    pbar = tqdm(total=len(df), desc="Processing test cases")

    for idx, row in df.iterrows():
        try:
            generated = qa_pipeline(row["Refined Question"], row["Context"])
            
            metrics = calculate_metrics(
                generated=generated,
                reference=row["Answer"]
            )

            log_interaction(
                context=row["Context"],
                question=row["Refined Question"],
                generated=generated,
                reference=row["Answer"],
                metrics=metrics
            )

            result_row = pd.DataFrame([{
                "question": row["Refined Question"],
                "context": row["Context"],
                "generated_answer": generated,
                "reference_answer": row["Answer"],
                **metrics
            }])
            
            result_row.to_csv(RESULTS_FILE, mode="a", header=False, index=False)

            pbar.update(1)
            pbar.set_postfix({"Processed": f"{idx+1}/{len(df)}", "Score": f"{metrics['final_score']:.2f}"})
            
        except Exception as e:
            error_msg = f"Error processing case {idx}: {str(e)}"
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "error": error_msg,
                    "context": row.get("Context", ""),
                    "question": row.get("Refined Question", "")
                }) + "\n")
            continue

    pbar.close()
    print(f"\n✅ Processing complete! Results saved to {RESULTS_FILE}. Logs in {LOG_FILE}.")

if __name__ == "__main__":
    process_test_cases()
    
    final_df = pd.read_csv(RESULTS_FILE)
    print("\nFinal Metrics Summary:")
    print(final_df[['exact_match', 'cosine_similarity', 'rouge_score', 'bert_similarity', 'bert_score_precision' , 'bert_score_recall', 'bert_score_f1', 'precision_score', 'final_score']].mean())