import os
import glob
import re
import sys
import requests
import uuid
import json
import tqdm
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# For vectorization and embedding
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.schema import Document
import chromadb

# Load environment variables
load_dotenv()

class MedlinePlusScraper:
    """Class to handle scraping of MedlinePlus encyclopedia articles."""
    
    BASE_URL = "https://medlineplus.gov/ency/"
    
    def __init__(self, output_dir="medlineplus_diseases"):
        """
        Initialize the scraper with session for connection reuse.
        
        Args:
            output_dir: Directory to save the disease text files
        """
        self.session = requests.Session()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
    
    def retrieve_webpage(self, url: str) -> Optional[str]:
        """
        Retrieve HTML content from a URL.
        
        Args:
            url: The URL to retrieve content from
            
        Returns:
            HTML content as string or None if retrieval failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.text
        except requests.RequestException as e:
            print(f"Error retrieving {url}: {e}")
            return None
    
    def parse_article_content(self, html: str) -> Dict[str, str]:
        """
        Extract article content from HTML.
        
        Args:
            html: HTML content to parse
            
        Returns:
            Dictionary with article sections and their content
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Extracting article title
            title_tag = soup.find("h1", class_="with-also", itemprop="name")
            article_title = title_tag.get_text(strip=True) if title_tag else "Title not found"
            
            extracted_text = {"Title": article_title}
            
            # Extract all sections dynamically
            for section in soup.find_all("div", class_="section"):
                title_div = section.find("div", class_="section-title")
                body_div = section.find("div", class_="section-body")
                
                if title_div and body_div:
                    section_title = title_div.get_text(strip=True)
                    section_content = body_div.get_text(" ", strip=True)
                    extracted_text[section_title] = section_content
            
            return extracted_text
        except Exception as e:
            print(f"Error parsing article content: {e}")
            return {"Error": f"Failed to parse content: {str(e)}"}
    
    def create_safe_filename(self, title: str) -> str:
        """
        Create a safe filename from the article title.
        
        Args:
            title: The article title
            
        Returns:
            A safe filename without invalid characters
        """
        # Remove invalid filename characters
        safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
        # Replace spaces and multiple non-alphanumeric chars with underscore
        safe_title = re.sub(r'\s+', "_", safe_title)
        safe_title = re.sub(r'[^a-zA-Z0-9_.-]', "", safe_title)
        
        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Truncate if filename is too long (Windows has 260 char path limit)
        max_length = 200  # Leave room for directory, extension, and timestamp
        if len(safe_title) > max_length:
            safe_title = safe_title[:max_length]
            
        return f"{safe_title}_{timestamp}.txt"
    
    def save_to_file(self, content: Dict[str, str], url: str) -> str:
        """
        Save the extracted content to a text file.
        
        Args:
            content: Dictionary with article sections and their content
            url: Source URL of the content
            
        Returns:
            Path to the saved file or error message
        """
        try:
            title = content.get("Title", "Unknown_Disease")
            filename = self.create_safe_filename(title)
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(f"Source: {url}\n")
                file.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write each section
                for section, text in content.items():
                    file.write(f"{section}\n")
                    file.write(f"{text}\n\n")
            
            return filepath
        except Exception as e:
            print(f"Error saving file: {e}")
            return f"Error: {str(e)}"
    
    def find_encyclopedia_articles(self, letter: str) -> List[str]:
        """
        Find all article links for a given letter in the encyclopedia.
        
        Args:
            letter: Single letter to retrieve articles for
            
        Returns:
            List of article URLs
        """
        # Validate input
        if not letter or len(letter.strip()) != 1 or not letter.strip().isalpha():
            raise ValueError("Please provide a single alphabetical character")
            
        letter = letter.strip().upper()
        url = f"{self.BASE_URL}encyclopedia_{letter}.htm"
        html = self.retrieve_webpage(url)
        
        if not html:
            return []
        
        soup = BeautifulSoup(html, "html.parser")
        article_links = []
        
        # Find all article links
        for li in soup.select("#mplus-content li"):
            if not li.get("class"):  # Ensure <li> has no class
                a_tag = li.find("a", href=True)
                if a_tag and a_tag["href"].startswith("article/"):
                    article_links.append(self.BASE_URL + a_tag["href"])
        
        return article_links
        
    def scrape_and_save_articles(self, letter: str) -> None:
        """
        Main function to scrape articles for a given letter and save to files.
        
        Args:
            letter: Single letter to retrieve articles for
        """
        try:
            article_links = self.find_encyclopedia_articles(letter)
            
            if not article_links:
                print(f"No articles found for letter '{letter}'.")
                return
            
            print(f"Found {len(article_links)} articles for letter '{letter}'.")
            successful_saves = 0
            
            for link in article_links:
                print(f"\nProcessing: {link}")
                html = self.retrieve_webpage(link)
                
                if html:
                    extracted_text = self.parse_article_content(html)
                    
                    # Save to file
                    saved_path = self.save_to_file(extracted_text, link)
                    if not saved_path.startswith("Error"):
                        print(f"✓ Saved to: {os.path.basename(saved_path)}")
                        successful_saves += 1
                    else:
                        print(f"✗ Failed to save: {saved_path}")
                else:
                    print(f"✗ Could not retrieve content from {link}")
            
            print(f"\nSummary: Successfully saved {successful_saves} out of {len(article_links)} articles.")
            print(f"Files are located in: {os.path.abspath(self.output_dir)}")
                    
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


class MedlinePlusVectorizer:
    """Process MedlinePlus data into a ChromaDB vector database and implement RAG pipeline."""
    
    def __init__(
        self, 
        input_dir="medlineplus_diseases", 
        chunk_size=1000, 
        chunk_overlap=200,
        collection_name="medlineplus_collection",
        initialize_embeddings=True  # Add flag to control initialization
    ):
        """
        Initialize the vectorizer.
        
        Args:
            input_dir: Directory containing scraped MedlinePlus files
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            collection_name: Name for the ChromaDB collection
            initialize_embeddings: Whether to initialize embeddings (can be disabled for testing)
        """
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        
        # Initialize ChromaDB client for document processing
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use HuggingFaceEmbeddings for LangChain compatibility
        if initialize_embeddings:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
    
    def combine_files(self) -> str:
        """
        Combine all text files in the input directory into a single string.
        
        Returns:
            Combined text from all files
        """
        print(f"Combining files from {self.input_dir}...")
        combined_text = ""
        file_count = 0
        
        # Get all .txt files in the input directory
        file_paths = glob.glob(os.path.join(self.input_dir, "*.txt"))
        
        for file_path in tqdm.tqdm(file_paths):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Add file separator for clarity
                    combined_text += f"\n--- START OF DOCUMENT: {os.path.basename(file_path)} ---\n\n"
                    combined_text += content
                    combined_text += f"\n--- END OF DOCUMENT: {os.path.basename(file_path)} ---\n\n"
                    
                    file_count += 1
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        
        print(f"Successfully combined {file_count} files.")
        return combined_text
    
    def create_chunks(self, text: str) -> List[Document]:
        """
        Split the combined text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            text: The combined text to be chunked
            
        Returns:
            List of LangChain Document objects
        """
        print(f"Creating chunks with size {self.chunk_size} and overlap {self.chunk_overlap}...")
        
        # Split the text into documents
        documents = self.text_splitter.create_documents([text])
        
        # Add metadata to each document
        for doc in documents:
            doc.metadata["source"] = "combined_text"
            doc.metadata["chunk_id"] = str(uuid.uuid4())
        
        print(f"Created {len(documents)} chunks from the combined text.")
        return documents
    
    def create_vector_db(self, documents: List[Document]) -> None:
        """
        Create a vector database from the documents using LangChain's Chroma.
        
        Args:
            documents: List of LangChain Document objects
        """
        print(f"Creating vector database with collection name: {self.collection_name}...")
        
        try:
            # Use LangChain's Chroma to create and store the vector database
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory="./chroma_db",
                collection_name=self.collection_name
            )
            
            # Persist the vector store
            vector_store.persist()
            
            print(f"Successfully created vector database with {len(documents)} entries.")
            print(f"Database stored at: {os.path.abspath('./chroma_db')}")
            
        except Exception as e:
            print(f"Error creating vector database: {e}")
    
    def process(self) -> None:
        """Main processing function to execute the entire pipeline."""
        try:
            # Step 1: Combine all files
            combined_text = self.combine_files()
            
            # Step 2: Create chunks from combined text
            documents = self.create_chunks(combined_text)
            
            # Step 3 & 4: Create embeddings and store in ChromaDB
            self.create_vector_db(documents)
            
            print("Processing completed successfully!")
        except Exception as e:
            print(f"An error occurred during processing: {e}")
    
    def initialize_rag_pipeline(self):
        print("Initializing RAG pipeline...")
        """Initialize the RAG pipeline using LangChain."""
        try:
            # Load the vector store using LangChain's Chroma
            vector_store = Chroma(
                persist_directory="./chroma_db",
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
            
            # Initialize the Mistral model
            mistral_llm = self.initialize_mistral_model()
            
            # Define the prompt template for RAG
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "You are a medical assistant. Answer the user's question using ONLY the provided context. "
                    "If unsure, say so. Always explain medical terms in simple language.\n\n"
                    "Context: {context}\n\n"
                    "Question: {question}"
                )
            )
            
            # Initialize the RetrievalQA chain
            rag_pipeline = RetrievalQA.from_chain_type(
                llm=mistral_llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            return rag_pipeline
            
        except Exception as e:
            print(f"Error initializing RAG pipeline: {e}")
            raise e
    
    def query_with_rag(self, query_text: str) -> Tuple[str, str]:
        """Query the ChromaDB vector database and generate an answer using RAG."""
        try:
            # Initialize the RAG pipeline
            rag_pipeline = self.initialize_rag_pipeline()
            
            # Run the query through the RAG pipeline
            result = rag_pipeline.run(query_text)
            
            # Log the query and answer
            self._log_query(query_text, result)
            
            return result, ""  # Return the answer and empty context (context is already used in RAG)
                
        except Exception as e:
            print(f"Error querying the database or generating response: {e}")
            return f"Error: {e}", ""
    
    def initialize_mistral_model(self):
        """Initializes the Mistral model using LangChain."""
        print("Initializing Mistral model...")
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("Mistral API key is required. Please set the MISTRAL_API_KEY environment variable.")
        
        # Initialize the Mistral model with the API key
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,  # Lower temperature for more factual responses
            max_retries=2,     # Retry on API failures
            api_key=mistral_api_key  # Pass the API key directly
        )
        return llm
        
    def _log_query(self, query: str, answer: str) -> None:
        """Logs the query and answer into a JSON file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": query,
            "answer": answer
        }
        
        # Append the log entry to a JSON file
        log_file = "query_logs.json"
        try:
            # Try to read existing logs
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        print("Warning: Log file is corrupted. Starting a new log.")
                        logs = []
            else:
                logs = []
            
            # Append new log entry
            logs.append(log_entry)
            
            # Write back to the file
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            print(f"Error logging query: {e}")


def test_api_key():
    """Test if the Mistral API key is available."""
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        print("✓ Mistral API key is available")
        return True
    else:
        print("✗ Mistral API key is missing")
        return False


def test_scraper(letter='Z', limit=3):
    """Test the MedlinePlus scraper with a single letter."""
    print("\n=== TESTING MEDLINEPLUS SCRAPER ===\n")
    
    test_dir = "test_medlineplus_output"
    
    # Initialize the scraper with test directory
    scraper = MedlinePlusScraper(output_dir=test_dir)
    
    # Find articles for the letter
    articles = scraper.find_encyclopedia_articles(letter)
    
    if not articles:
        print(f"No articles found for letter '{letter}'.")
        return
    
    print(f"Found {len(articles)} articles for letter '{letter}'.")
    print(f"Testing with first {limit} articles...")
    
    # Process a limited number of articles
    for i, link in enumerate(articles[:limit]):
        print(f"\nProcessing article {i+1}/{limit}: {link}")
        html = scraper.retrieve_webpage(link)
        
        if html:
            content = scraper.parse_article_content(html)
            print(f"Extracted sections: {list(content.keys())}")
            
            saved_path = scraper.save_to_file(content, link)
            if not saved_path.startswith("Error"):
                print(f"✓ Saved to: {os.path.basename(saved_path)}")
            else:
                print(f"✗ Failed to save: {saved_path}")
        else:
            print(f"✗ Could not retrieve content from {link}")
    
    print(f"\nScraper test completed. Check {os.path.abspath(test_dir)} for output files.")
    return test_dir


def test_vectorizer(input_dir):
    """Test the MedlinePlus vectorizer with the given input directory."""
    print("\n=== TESTING MEDLINEPLUS VECTORIZER ===\n")
    
    # Initialize the vectorizer with test directory
    vectorizer = MedlinePlusVectorizer(
        input_dir=input_dir,
        chunk_size=1000, 
        chunk_overlap=200,
        collection_name="test_medlineplus_collection"
    )
    
    try:
        # Test combining files
        print("Testing file combination...")
        combined_text = vectorizer.combine_files()
        print(f"Combined text length: {len(combined_text)} characters")
        
        # Test chunking
        print("\nTesting text chunking...")
        documents = vectorizer.create_chunks(combined_text)
        print(f"Created {len(documents)} document chunks")
        
        # Test vector DB creation (optional - can be slow)
        user_input = input("\nCreate vector database? (y/n): ").strip().lower()
        if user_input == 'y':
            print("\nCreating vector database...")
            vectorizer.create_vector_db(documents)
            
            # Test querying (optional - requires Mistral API key)
            if test_api_key():
                user_input = input("\nTest querying with RAG? (y/n): ").strip().lower()
                if user_input == 'y':
                    query = input("Enter a medical question: ")
                    print("\nQuerying the RAG pipeline...")
                    answer, _ = vectorizer.query_with_rag(query)
                    print(f"\nQuestion: {query}")
                    print(f"Answer: {answer}")
        
        print("\nVectorizer test completed.")
        
    except Exception as e:
        print(f"Error during vectorizer testing: {e}")


def main():
    """Main function to run tests."""
    print("=== MEDLINEPLUS SCRAPER AND VECTORIZER TESTER ===\n")
    
    while True:
        print("\nSelect a test to run:")
        print("1. Test API key")
        print("2. Test scraper only")
        print("3. Test vectorizer with existing data")
        print("4. Run complete end-to-end test")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            test_api_key()
        
        elif choice == '2':
            letter = input("Enter a letter to scrape (default: Z): ").strip() or 'Z'
            limit = int(input("Enter max number of articles to scrape (default: 3): ").strip() or '3')
            test_scraper(letter, limit)
        
        elif choice == '3':
            input_dir = input("Enter the directory with MedlinePlus data: ").strip()
            if os.path.exists(input_dir):
                test_vectorizer(input_dir)
            else:
                print(f"Directory {input_dir} does not exist.")
        
        elif choice == '4':
            letter = input("Enter a letter to scrape (default: Z): ").strip() or 'Z'
            limit = int(input("Enter max number of articles to scrape (default: 3): ").strip() or '3')
            test_dir = test_scraper(letter, limit)
            if os.path.exists(test_dir):
                test_vectorizer(test_dir)
            
        elif choice == '5':
            print("Exiting...")
            break
        
        else:

            print("Invalid choice. Please enter a number between 1 and 5.")


if __name__ == "__main__":
    main()