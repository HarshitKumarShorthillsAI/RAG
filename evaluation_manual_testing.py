import os
import pandas as pd
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from dotenv import load_dotenv
from bert_score import score as bert_score

load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MANUAL_TEST_RESULTS_FILE = "manual_test_results.xlsx"
TEST_CASES_FILE = "manual_test_cases.xlsx"
VALID_MODELS = ["mistral-7b", "mistral-tiny", "gpt-4"]
MISTRAL_MODEL = "mistral-tiny"

# Initialize models
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Metric weights for final score calculation
METRIC_WEIGHTS = {
    "rouge_score": 0.25,
    "cosine_similarity": 0.25,
    "bert_score_f1": 0.5  # Using F1 as the main BERT Score metric
}

def validate_environment():
    """Validate that environment is properly set up"""
    issues = []
    
    # Check API key
    if not MISTRAL_API_KEY:
        issues.append("❌ MISTRAL_API_KEY not found in environment variables")
    
    # Check required files
    if not os.path.exists(TEST_CASES_FILE):
        issues.append(f"❌ Test cases file {TEST_CASES_FILE} not found")
    
    # Check required packages
    try:
        import sentence_transformers
        import rouge_score
        import bert_score
    except ImportError as e:
        issues.append(f"❌ Required package not installed: {str(e)}")
    
    if issues:
        print("\n".join(issues))
        print("\nEnvironment validation failed. Please fix the issues above.")
        return False
    
    print("✅ Environment validation passed!")
    return True

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

def qa_pipeline(question, context, model=MISTRAL_MODEL):
    """Query Mistral API with context, with exponential backoff retry handling."""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Modified to include context in the prompt
    payload = {
        "model": model,
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

def run_single_test_case(test_id):
    """Run a single test case by ID"""
    # Load test cases
    try:
        df = pd.read_excel(TEST_CASES_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Error loading test cases: {str(e)}")
        return None
    
    # Find test case by ID
    test_case = df[df["TEST CASE ID"] == test_id]
    if test_case.empty:
        print(f"❌ Test case with ID '{test_id}' not found.")
        return None
    
    test_case = test_case.iloc[0]
    
    # Extract test data
    question = test_case.get("TEST DATA")
    context = test_case.get("PRECONDITIONS")
    expected_result = test_case.get("EXPECTED RESULT")
    
    if not all([question, context, expected_result]):
        print("❌ Required test data missing. Ensure TEST DATA, PRECONDITIONS, and EXPECTED RESULT are filled.")
        return None
    
    print(f"\n🧪 Running test case {test_id}...")
    print(f"Question: {question}")
    print(f"Context: {context[:100]}...")
    
    # Run test
    start_time = time.time()
    generated = qa_pipeline(question, context)
    end_time = time.time()
    
    # Calculate metrics
    metrics = calculate_metrics(generated, expected_result)
    
    # Build result
    result = {
        "TEST CASE ID": test_id,
        "SECTION": test_case.get("SECTION"),
        "SUB-SECTION": test_case.get("SUB-SECTION"),
        "TEST CASE TITLE": test_case.get("TEST CASE TITLE"),
        "TEST DESCRIPTION": test_case.get("TEST DESCRIPTION"),
        "PRECONDITIONS": context,
        "TEST DATA": question,
        "TEST STEPS": test_case.get("TEST STEPS"),
        "EXPECTED RESULT": expected_result,
        "ACTUAL RESULT": generated,
        "STATUS": "PASS" if metrics["final_score"] >= 0.7 else "FAIL",
        "RESPONSE TIME": f"{(end_time - start_time):.2f}s",
        **metrics
    }
    
    print(f"Generated: {generated}")
    print(f"Metrics: ROUGE={metrics['rouge_score']:.4f}, Cosine={metrics['cosine_similarity']:.4f}, BERT-F1={metrics['bert_score_f1']:.4f}")
    print(f"Final Score: {metrics['final_score']:.4f}")
    print(f"Status: {result['STATUS']}")
    
    return result

def run_all_test_cases():
    """Run all test cases from the Excel file"""
    # Load test cases
    try:
        df = pd.read_excel(TEST_CASES_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Error loading test cases: {str(e)}")
        return None
    
    results = []
    
    print(f"🧪 Running {len(df)} test cases...")
    
    for _, test_case in df.iterrows():
        test_id = test_case["TEST CASE ID"]
        result = run_single_test_case(test_id)
        if result:
            results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_excel(MANUAL_TEST_RESULTS_FILE, index=False, engine="openpyxl")
    
    # Calculate summary
    passed = len(results_df[results_df["STATUS"] == "PASS"])
    failed = len(results_df[results_df["STATUS"] == "FAIL"])
    pass_rate = passed / len(results_df) * 100 if results_df.shape[0] > 0 else 0
    
    print(f"\n✅ Testing complete! Results saved to {MANUAL_TEST_RESULTS_FILE}")
    print(f"Summary: {passed} passed, {failed} failed ({pass_rate:.1f}% pass rate)")
    
    # Calculate average metrics
    metrics_cols = ['rouge_score', 'cosine_similarity', 'bert_score_f1', 'final_score']
    if not results_df.empty:
        avg_metrics = results_df[metrics_cols].mean()
        print("\nAverage Metrics:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results_df

def create_test_case_template():
    """Create template Excel file for test cases"""
    template_data = {
        "TEST CASE ID": ["TC001", "TC002", "TC003"],
        "SECTION": ["API", "Metrics", "Data Processing"],
        "SUB-SECTION": ["Authentication", "BERT Score", "Excel Handling"],
        "TEST CASE TITLE": [
            "Mistral API Authentication", 
            "Metric Calculation Accuracy", 
            "Excel File Loading"
        ],
        "TEST DESCRIPTION": [
            "Verify that authentication with Mistral API works correctly",
            "Verify the accuracy of metric calculations between generated and reference answers",
            "Test the Excel file loading functionality with various file formats"
        ],
        "PRECONDITIONS": [
            "Valid Mistral API key is set in environment",
            "Sample text: 'The quick brown fox jumps over the lazy dog.'",
            "Excel file exists in the correct format"
        ],
        "TEST DATA": [
            "Test question to query API",
            "Reference answer: 'The quick brown fox jumps over the lazy dog.'",
            "Path to test Excel file"
        ],
        "TEST STEPS": [
            "1. Call qa_pipeline with test question\n2. Verify response structure",
            "1. Calculate metrics between sample and reference text\n2. Verify score ranges",
            "1. Call load_test_cases with file path\n2. Verify dataframe output"
        ],
        "EXPECTED RESULT": [
            "API returns a valid response",
            "Metrics calculated correctly: ROUGE > 0.5, Cosine > 0.7, BERT > 0.8",
            "DataFrame loaded with expected columns and rows"
        ],
        "ACTUAL RESULT": ["", "", ""],
        "STATUS": ["Not Run", "Not Run", "Not Run"]
    }
    
    template_df = pd.DataFrame(template_data)
    template_df.to_excel(TEST_CASES_FILE, index=False, engine="openpyxl")
    print(f"✅ Test case template created: {TEST_CASES_FILE}")

def main_menu():
    """Display interactive menu for manual testing"""
    while True:
        print("\n" + "="*50)
        print("🧪 QA Evaluation Manual Testing Framework")
        print("="*50)
        print("1. Validate environment")
        print("2. Create test case template")
        print("3. Run a single test case")
        print("4. Run all test cases")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            validate_environment()
        
        elif choice == "2":
            create_test_case_template()
        
        elif choice == "3":
            test_id = input("Enter test case ID: ")
            result = run_single_test_case(test_id)
            if result:
                # Save this single result
                pd.DataFrame([result]).to_excel(
                    f"result_{test_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx", 
                    index=False, 
                    engine="openpyxl"
                )
        
        elif choice == "4":
            run_all_test_cases()
        
        elif choice == "5":
            print("Exiting program. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1-5.")

if __name__ == "__main__":
    main_menu()