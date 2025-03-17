import os
import csv
import time
import chromadb
from chromadb.utils import embedding_functions
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

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

# Generate specific questions based on the content
def generate_questions(content: str, disease_name: str) -> list[str]:
    """Generates specific questions based on the content and disease name."""
    questions = []
    sections = [
        "Causes", "Symptoms", "Treatment", "Prevention"
    ]
    
    for section in sections:
        if section in content:
            questions.append(f"What is the {section.lower()} of {disease_name}?")
    
    return questions

# Exponential backoff for rate limit handling
def exponential_backoff(retries, initial_delay=2):
    """Calculates the delay for exponential backoff."""
    return initial_delay * (2 ** (retries - 1))

# Refine questions using Mistral with exponential backoff
def refine_questions(questions: list[str], mistral_llm) -> list[str]:
    """Refines the questions using Mistral with exponential backoff."""
    refined_questions = []
    for question in questions:
        prompt = (
            "You are a medical assistant. Refine the following question to make it more specific and clear. "
            "Ensure the question is easy to understand and directly related to the disease. "
            "Only provide the refined question, without any additional explanations or metadata.\n\n"
            f"Original Question: {question}\n\n"
            "Refined Question:"
        )
        
        retries = 3  # Number of retries
        for i in range(retries):
            try:
                response = mistral_llm.invoke(prompt)
                refined_question = response.content.split("\n")[0].strip()
                refined_questions.append(refined_question)
                print(f"Refined question: {refined_question}")
                break  # Exit the retry loop on success
            except Exception as e:
                if "rate limit" in str(e).lower():
                    delay = exponential_backoff(i + 1)
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Error refining question: {question}. Error: {e}")
                    refined_questions.append(question)  # Fallback to the original question
                    break  # Exit the retry loop on other errors
        
        # Add a delay to avoid hitting the rate limit
        time.sleep(2)  # Adjust the delay as needed
    
    return refined_questions

# Generate an answer using Mistral and ChromaDB
def generate_answer(refined_question: str, mistral_llm, chroma_collection) -> str:
    """Generates an answer using Mistral and ChromaDB."""
    # Query ChromaDB for relevant context
    results = chroma_collection.query(
        query_texts=[refined_question],
        n_results=3  # Retrieve top 3 relevant chunks
    )
    
    # Combine the top chunks into context
    context = "\n\n".join(results['documents'][0])
    
    # Generate answer using Mistral
    prompt = (
        "You are a medical assistant. Answer the user's question using ONLY the provided context. "
        "If unsure, say so. Always explain medical terms in simple language.\n\n"
        f"Context: {context}\n\n"
        f"Question: {refined_question}"
    )
    
    retries = 3  # Number of retries
    for i in range(retries):
        try:
            response = mistral_llm.invoke(prompt)
            return response.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                delay = exponential_backoff(i + 1)
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Error generating answer: {e}")
                return "Error: Could not generate answer."
    
    return "Error: Max retries exceeded."

# Process each file in the directory
def process_files(directory: str, output_csv: str):
    """Processes each file in the directory, generates questions, refines them, and stores them in a CSV file."""
    # Initialize Mistral model
    mistral_llm = initialize_mistral_model()
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_collection(
        name="medlineplus_collection",  # Replace with your collection name
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    
    # Open the CSV file for writing
    with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write the header row
        writer.writerow(["Refined Question", "Answer"])  # Store both refined questions and answers
        
        # Iterate over the first 3 files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    
                    # Extract the disease name (title) from the filename
                    disease_name = extract_disease_name(filename)
                    
                    # Generate specific questions
                    questions = generate_questions(content, disease_name)
                    
                    # Refine questions using Mistral
                    refined_questions = refine_questions(questions, mistral_llm)
                    
                    # Generate answers for each refined question
                    for refined_question in refined_questions:
                        # Generate answer using Mistral and ChromaDB
                        answer = generate_answer(refined_question, mistral_llm, chroma_collection)
                        
                        # Write refined question and answer to CSV
                        writer.writerow([refined_question, answer])
                        print(f"Refined question: {refined_question}")
                        print(f"Answer: {answer}\n")
                    
                    print(f"Processed file: {filename}")

# Generate context for a refined question using ChromaDB
def generate_context(refined_question: str, chroma_collection) -> str:
    """Generates context for a refined question using ChromaDB."""
    # Query ChromaDB for relevant context
    results = chroma_collection.query(
        query_texts=[refined_question],
        n_results=3  # Retrieve top 3 relevant chunks
    )
    
    # Combine the top chunks into context
    context = "\n\n".join(results['documents'][0])
    return context

# Generate context for each refined question in the existing CSV file
# Generate context for each refined question in the existing CSV file
def generate_context_file(input_csv: str, context_csv: str):
    """Generates a context file for each refined question in the existing CSV file."""
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
        context_writer.writerow(["Refined Question", "Answer", "Context"])
        
        # Skip the header row in the input CSV
        next(reader)
        
        # Iterate over each row in the input CSV
        for row in reader:
            refined_question = row[0]  # First column contains the refined question
            answer = row[1]  # Second column contains the answer
            
            # Generate context for the refined question
            context = generate_context(refined_question, chroma_collection)
            
            # Write the refined question, answer, and context to the context CSV
            context_writer.writerow([refined_question, answer, context])
            print(f"Refined question: {refined_question}")
            print(f"Answer: {answer}")
            print(f"Context: {context}\n")

# Main function
def main():
    # Directory containing the files
    input_directory = "medlineplus_diseases"  # Replace with your directory path
    # Output CSV files
    output_csv = "refined_questions_with_answers.csv"
    context_csv = "refined_questions_with_context.csv"
    
    # Display menu options
    print("Choose an option:")
    print("1. Store questions with refined answers")
    print("2. Generate and store context for refined questions")
    
    # Get user input
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Option 1: Store questions with refined answers
        process_files(input_directory, output_csv)
        print(f"Refined questions and answers saved to {output_csv}")
    elif choice == "2":
        # Option 2: Generate and store context for refined questions
        if not os.path.exists(output_csv):
            print(f"Error: {output_csv} does not exist. Please run Option 1 first.")
        else:
            generate_context_file(output_csv, context_csv)
            print(f"Contexts saved to {context_csv}")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()