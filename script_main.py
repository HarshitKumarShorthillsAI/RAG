import os
import csv
import time
import chromadb
from chromadb.utils import embedding_functions
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import glob
import uuid
import tqdm
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple
from Vectorizer import MedlinePlusVectorizer
import pandas as pd  # Import pandas for Excel file handling
import backoff  # For retry mechanism
import requests  # For handling HTTP errors
import logging  # For better error logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define common API errors to handle
class RateLimitError(Exception):
    """Exception raised when API rate limit is exceeded."""
    pass

# Function to determine if an exception is due to rate limiting
def is_rate_limit_error(exception):
    """Check if the exception is related to rate limiting."""
    if isinstance(exception, requests.exceptions.HTTPError):
        # Check for common HTTP status codes for rate limiting
        if exception.response.status_code in [429, 503]:
            return True
    
    # Check error message for rate limit indicators
    error_msg = str(exception).lower()
    rate_limit_indicators = ["rate limit", "too many requests", "quota exceeded", "throttle"]
    return any(indicator in error_msg for indicator in rate_limit_indicators)

# Initialize Mistral model
def initialize_mistral_model():
    """Initializes the Mistral model using LangChain."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("Mistral API key not found. Please set the MISTRAL_API_KEY environment variable.")
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,  # Lower temperature for more factual responses
        max_retries=3,     # Increase retry attempts on API failures
        api_key=mistral_api_key,  # Pass the API key directly
        request_timeout=60  # Set timeout to 60 seconds
    )
    return llm

# Extract the title (disease name) from the filename
def extract_disease_name(filename: str) -> str:
    """Extracts the disease name (title) from the filename."""
    # Remove the file extension and split by underscores
    filename_without_ext = os.path.splitext(filename)[0]
    parts = filename_without_ext.split("_")
    
    # Extract the disease name (all parts before the last numeric part)
    disease_name_parts = []
    for part in parts:
        if part.isdigit():
            break
        disease_name_parts.append(part)
    
    # Join the parts with spaces to form the disease name
    disease_name = " ".join(disease_name_parts)
    return disease_name

# Hardcoded questions for each disease
def generate_questions(disease_name: str) -> list[str]:
    """Generates hardcoded questions for the disease."""
    questions = [
        f"What are the symptoms of {disease_name}?",
        f"What are the prevention for {disease_name}?",
        f"What are the treatment for {disease_name}?",
        f"What are the causes of {disease_name}?"
    ]
    return questions

# Custom backoff handler for logging
def backoff_handler(details):
    """Handler called on backoff."""
    exception = details["exception"]
    tries = details["tries"]
    wait = details["wait"]
    
    if is_rate_limit_error(exception):
        logger.warning(f"Rate limit exceeded! Retry {tries} in {wait:.1f} seconds...")
    else:
        logger.warning(f"Error: {exception}. Retry {tries} in {wait:.1f} seconds...")

# Retry mechanism for handling timeouts or temporary failures with special handling for rate limits
@backoff.on_exception(
    backoff.expo,
    Exception,  # Catch all exceptions but handle rate limits differently
    max_tries=10,  # Maximum number of retry attempts
    max_time=600,  # Maximum total time to keep retrying (10 minutes)
    giveup=lambda e: not (isinstance(e, Exception) and (is_rate_limit_error(e) or isinstance(e, TimeoutError))),
    on_backoff=backoff_handler,
    jitter=backoff.full_jitter,  # Add randomness to backoff times
    factor=2.5  # Exponential backoff factor (longer waits for rate limits)
)
def generate_answer_with_retry(question: str, vectorizer) -> str:
    """Generates an answer using the RAG pipeline with retry mechanism."""
    try:
        # Query the RAG pipeline
        answer, _ = vectorizer.query_with_rag(question)
        return answer
    except Exception as e:
        if is_rate_limit_error(e):
            logger.error(f"Rate limit exceeded while generating answer for: {question}")
            # Sleep for a longer time on rate limit errors
            time.sleep(30)  # Extra sleep for rate limit cooling
            raise RateLimitError(f"Rate limit exceeded: {str(e)}")
        else:
            logger.error(f"Error generating answer: {e}. Retrying...")
            raise  # Re-raise to trigger backoff retry

# Generate context for a question using ChromaDB
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=10,
    max_time=600,
    giveup=lambda e: not (isinstance(e, Exception) and (is_rate_limit_error(e) or isinstance(e, TimeoutError))),
    on_backoff=backoff_handler,
    jitter=backoff.full_jitter,
    factor=2.5
)
def generate_context_with_retry(question: str, chroma_collection) -> str:
    """Generates context for a question using ChromaDB with retry mechanism."""
    try:
        # Query ChromaDB for relevant context
        results = chroma_collection.query(
            query_texts=[question],
            n_results=3  # Retrieve top 3 relevant chunks
        )
        
        # Combine the top chunks into context
        context = "\n\n".join(results['documents'][0])
        return context
    except Exception as e:
        if is_rate_limit_error(e):
            logger.error(f"Rate limit exceeded while generating context for: {question}")
            # Sleep for a longer time on rate limit errors
            time.sleep(30)  # Extra sleep for rate limit cooling
            raise RateLimitError(f"Rate limit exceeded: {str(e)}")
        else:
            logger.error(f"Error generating context: {e}. Retrying...")
            raise  # Re-raise to trigger backoff retry

# Process each file in the directory
def process_files(directory: str, output_excel: str):
    """Processes all files in the directory, generates hardcoded questions, and stores answers in an Excel file."""
    # Initialize Mistral model
    mistral_llm = initialize_mistral_model()
    
    # Initialize MedlinePlusVectorizer for RAG pipeline
    vectorizer = MedlinePlusVectorizer(input_dir=directory)
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection(
        name="medlineplus_collection",  # Replace with your collection name
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Get all text files in the directory
    text_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    total_files = len(text_files)
    
    logger.info(f"Found {total_files} text files to process")
    
    # Create a DataFrame to store all the data
    # Check if excel file already exists to continue from where we left off
    if os.path.exists(output_excel):
        try:
            df = pd.read_excel(output_excel)
            logger.info(f"Loaded existing data from {output_excel} with {len(df)} records")
            
            # Get list of already processed diseases from the questions
            processed_diseases = set()
            for question in df["Question"]:
                if "What are the symptoms of " in question:
                    disease = question.replace("What are the symptoms of ", "").replace("?", "")
                    processed_diseases.add(disease)
            
            logger.info(f"Found {len(processed_diseases)} already processed diseases: {', '.join(list(processed_diseases)[:5])}{'...' if len(processed_diseases) > 5 else ''}")
        except Exception as e:
            logger.error(f"Error reading existing Excel file: {e}. Starting with empty DataFrame.")
            df = pd.DataFrame(columns=["Question", "Answer", "Context"])
    else:
        df = pd.DataFrame(columns=["Question", "Answer", "Context"])
    
    processed_files = 0
    
    # Process all files
    for i, filename in enumerate(text_files):
        filepath = os.path.join(directory, filename)
        
        # Extract the disease name from the filename
        disease_name = extract_disease_name(filename)
        
        # Skip if this disease has already been processed
        if "Question" in df.columns and any(f"What are the symptoms of {disease_name}?" in q for q in df["Question"].tolist()):
            logger.info(f"Skipping already processed disease: {disease_name} ({i+1}/{total_files})")
            processed_files += 1
            continue
        
        logger.info(f"Processing disease: {disease_name} ({i+1}/{total_files})")
        
        # Create a temporary list to store data for this file
        file_data = []
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()
                
                # Generate hardcoded questions
                questions = generate_questions(disease_name)
                
                # Generate answers and context for each hardcoded question using RAG pipeline
                for question in questions:
                    logger.info(f"  Question: {question}")
                    
                    # Generate answer using RAG pipeline with retry
                    try:
                        answer = generate_answer_with_retry(question, vectorizer)
                        logger.info(f"  Answer generated successfully")
                    except Exception as e:
                        logger.error(f"  Failed to generate answer after multiple retries: {e}")
                        answer = "Error: Could not generate answer after multiple attempts."
                    
                    # Generate context using ChromaDB with retry
                    try:
                        context = generate_context_with_retry(question, chroma_collection)
                        logger.info(f"  Context retrieved successfully")
                    except Exception as e:
                        logger.error(f"  Failed to retrieve context after multiple retries: {e}")
                        context = "Error: Could not retrieve context after multiple attempts."
                    
                    # Append the data to the temporary list
                    file_data.append({"Question": question, "Answer": answer, "Context": context})
            
            # Append to the DataFrame and save immediately after each disease
            if file_data:
                temp_df = pd.DataFrame(file_data)
                df = pd.concat([df, temp_df], ignore_index=True)
                
                # Save the DataFrame to an Excel file after each disease
                df.to_excel(output_excel, index=False)
                logger.info(f"✓ Updated data saved to {output_excel} (Total records: {len(df)})")
            
            processed_files += 1
            logger.info(f"Processed file: {filename} ({processed_files}/{total_files})")
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            # Try to save what we have so far even if this file fails
            if not df.empty:
                df.to_excel(output_excel, index=False)
                logger.info(f"✓ Saved current progress to {output_excel} despite error")
    
    # Final save to ensure all data is written
    if not df.empty:
        df.to_excel(output_excel, index=False)
        logger.info(f"✓ All questions, answers, and contexts saved to {output_excel}")
        logger.info(f"Processed {processed_files} out of {total_files} files")

# Main function
def main():
    # Directory containing the files
    input_directory = "medlineplus_diseases"  # Replace with your directory path
    # Output Excel file
    output_excel = "questions_with_answers_and_context.xlsx"
    
    # Display menu options
    print("Choose an option:")
    print("1. Store questions with answers and context")
    
    # Get user input
    choice = input("Enter your choice (1): ")
    
    if choice == "1":
        # Option 1: Store questions with answers and context
        process_files(input_directory, output_excel)
        logger.info(f"All processing completed. Final data saved to {output_excel}")
    else:
        logger.warning("Invalid choice. Please enter 1.")

if __name__ == "__main__":
    main()