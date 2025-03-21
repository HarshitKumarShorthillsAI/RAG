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

# Load environment variables
load_dotenv()

# Initialize Mistral model
def initialize_mistral_model():
    """Initializes the Mistral model using LangChain."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("Mistral API key not found. Please set the MISTRAL_API_KEY environment variable.")
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.2,  # Lower temperature for more factual responses
        max_retries=2,     # Retry on API failures
        api_key=mistral_api_key  # Pass the API key directly
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
        f"How can {disease_name} be prevented?",
        f"What are the treatments for {disease_name}?",
        f"What are the causes of {disease_name}?"
    ]
    return questions

# Generate an answer using the RAG pipeline
def generate_answer(question: str, vectorizer) -> str:
    """Generates an answer using the RAG pipeline."""
    try:
        # Query the RAG pipeline
        answer, _ = vectorizer.query_with_rag(question)
        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error: Could not generate answer."

# Process each file in the directory
def process_files(directory: str, output_csv: str):
    """Processes each file in the directory, generates hardcoded questions, and stores answers in a CSV file."""
    # Initialize Mistral model
    mistral_llm = initialize_mistral_model()
    
    # Initialize MedlinePlusVectorizer for RAG pipeline
    vectorizer = MedlinePlusVectorizer(input_dir=directory)
    
    # Open the CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(["Question", "Answer"])  # Store both questions and answers
        
        # Iterate over the first 3 files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    
                    # Extract the disease name (title) from the filename
                    disease_name = extract_disease_name(filename)
                    
                    # Generate hardcoded questions
                    questions = generate_questions(disease_name)
                    
                    # Generate answers for each hardcoded question using RAG pipeline
                    for question in questions:
                        # Generate answer using RAG pipeline
                        answer = generate_answer(question, vectorizer)
                        
                        # Write question and answer to CSV
                        writer.writerow([question, answer])
                        print(f"Question: {question}")
                        print(f"Answer: {answer}\n")
                    
                    print(f"Processed file: {filename}")

# Generate context for a question using ChromaDB
def generate_context(question: str, chroma_collection) -> str:
    """Generates context for a question using ChromaDB."""
    # Query ChromaDB for relevant context
    results = chroma_collection.query(
        query_texts=[question],
        n_results=3  # Retrieve top 3 relevant chunks
    )
    
    # Combine the top chunks into context
    context = "\n\n".join(results['documents'][0])
    return context

# Generate context for each question in the existing CSV file
def generate_context_file(input_csv: str, context_csv: str):
    """Generates a context file for each question in the existing CSV file."""
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection(
        name="medlineplus_collection",  # Replace with your collection name
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Open the input CSV file for reading and the context CSV file for writing
    with open(input_csv, mode="r", newline="", encoding="utf-8") as input_file, \
         open(context_csv, mode="w", newline="", encoding="utf-8") as context_file:
        reader = csv.reader(input_file)
        context_writer = csv.writer(context_file)
        
        # Write the header row for the context CSV
        context_writer.writerow(["Question", "Answer", "Context"])
        
        # Skip the header row in the input CSV
        next(reader)
        
        # Iterate over each row in the input CSV
        for row in reader:
            question = row[0]  # First column contains the question
            answer = row[1]  # Second column contains the answer
            
            # Generate context for the question
            context = generate_context(question, chroma_collection)
            
            # Write the question, answer, and context to the context CSV
            context_writer.writerow([question, answer, context])
            print(f"Question: {question}")
            print(f"Answer: {answer}")
            print(f"Context: {context}\n")

# Main function
def main():
    # Directory containing the files
    input_directory = "medlineplus_diseases"  # Replace with your directory path
    # Output CSV files
    output_csv = "questions_with_answers2222.csv"
    context_csv = "questions_with_context2222.csv"
    
    # Display menu options
    print("Choose an option:")
    print("1. Store questions with answers")
    print("2. Generate and store context for questions")
    
    # Get user input
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Option 1: Store questions with answers
        process_files(input_directory, output_csv)
        print(f"Questions and answers saved to {output_csv}")
    elif choice == "2":
        # Option 2: Generate and store context for questions
        if not os.path.exists(output_csv):
            print(f"Error: {output_csv} does not exist. Please run Option 1 first.")
        else:
            generate_context_file(output_csv, context_csv)
            print(f"Contexts saved to {context_csv}")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()