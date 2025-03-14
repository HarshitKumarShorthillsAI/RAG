
import os
import glob
import re
import sys
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.utils import embedding_functions
import uuid
import tqdm
from datetime import datetime
import torch
from transformers import pipeline

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
        try:
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
        except Exception as e:
            print(f"Error finding articles: {e}")
            return []
    
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
    """Process MedlinePlus data into a ChromaDB vector database."""
    
    def __init__(
        self, 
        input_dir="medlineplus_diseases", 
        chunk_size=1000, 
        chunk_overlap=200,
        collection_name="medlineplus_collection"
    ):
        """
        Initialize the vectorizer.
        
        Args:
            input_dir: Directory containing scraped MedlinePlus files
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            collection_name: Name for the ChromaDB collection
        """
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use the default embedding function (all-MiniLM-L6-v2)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
    
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
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """
        Split the combined text into overlapping chunks.
        
        Args:
            text: The combined text to be chunked
            
        Returns:
            List of dictionaries with chunk info (id, text, metadata)
        """
        print(f"Creating chunks with size {self.chunk_size} and overlap {self.chunk_overlap}...")
        chunks = []
        
        # Split into documents based on the separator
        documents = re.split(r'--- START OF DOCUMENT: (.+?) ---', text)
        
        # Skip the first element which is empty
        documents = documents[1:]
        
        # Process documents in pairs (filename, content)
        for i in range(0, len(documents), 2):
            if i+1 < len(documents):
                filename = documents[i].strip()
                content = documents[i+1]
                
                # Remove the END OF DOCUMENT marker
                content = re.sub(r'--- END OF DOCUMENT: .+? ---', '', content).strip()
                
                # Split the document content into chunks
                start_idx = 0
                chunk_id = 0
                
                while start_idx < len(content):
                    # Extract chunk with specified size
                    end_idx = min(start_idx + self.chunk_size, len(content))
                    chunk_text = content[start_idx:end_idx]
                    
                    # Create metadata for the chunk
                    metadata = {
                        "source": filename,
                        "chunk_id": chunk_id,
                        "start_char": start_idx,
                        "end_char": end_idx
                    }
                    
                    # Extract section title if available
                    section_match = re.search(r'^([A-Za-z\s]+)\n', chunk_text)
                    if section_match:
                        metadata["section"] = section_match.group(1).strip()
                    
                    # Create chunk document
                    chunk_doc = {
                        "id": f"{filename}_chunk_{chunk_id}_{uuid.uuid4().hex[:8]}",
                        "text": chunk_text,
                        "metadata": metadata
                    }
                    
                    chunks.append(chunk_doc)
                    chunk_id += 1
                    
                    # Move start position for next chunk, considering overlap
                    start_idx += (self.chunk_size - self.chunk_overlap)
                    
                    # Ensure we're not starting with whitespace
                    while start_idx < len(content) and content[start_idx].isspace():
                        start_idx += 1
        
        print(f"Created {len(chunks)} chunks from the combined text.")
        return chunks
    
    def create_vector_db(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Create a vector database from the chunks using ChromaDB.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
        """
        print(f"Creating vector database with collection name: {self.collection_name}...")
        
        # Get or create collection
        try:
            # Try to get existing collection or create a new one
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Prepare data for batch addition
            ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                end_idx = min(i + batch_size, len(chunks))
                print(f"Adding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}...")
                
                collection.add(
                    ids=ids[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx]
                )
            
            print(f"Successfully created vector database with {len(chunks)} entries.")
            print(f"Database stored at: {os.path.abspath('./chroma_db')}")
            
        except Exception as e:
            print(f"Error creating vector database: {e}")
    
    def process(self) -> None:
        """Main processing function to execute the entire pipeline."""
        try:
            # Step 1: Combine all files
            combined_text = self.combine_files()
            
            # Step 2: Create chunks from combined text
            chunks = self.create_chunks(combined_text)
            
            # Step 3 & 4: Create embeddings and store in ChromaDB
            self.create_vector_db(chunks)
            
            print("Processing completed successfully!")
        except Exception as e:
            print(f"An error occurred during processing: {e}")
    
    def query_with_llama(self, query_text: str, n_results: int = 5) -> None:
        """
        Query the ChromaDB vector database and generate an answer using LLaMA.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
        """
        try:
            # Get the ChromaDB collection
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Query the database for relevant chunks
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Combine the top chunks into context
            context = "\n\n".join(results['documents'][0])
            
            # Initialize the LLaMA model
            llama_pipeline = self.initialize_llama_model()
            
            # Generate an answer using LLaMA
            answer = self.generate_llama_response(query_text, context, llama_pipeline)
            
            # Print the results
            print("\n=== Generated Answer using llama Model ===")
            print(answer)
            print("\n=== Supporting Context ===")
            print(context[:1000] + "...")  # Show first 1000 chars of context
            
        except Exception as e:
            print(f"Error querying the database or generating response: {e}")
    
    def initialize_llama_model(self):
        """Initializes the LLaMA 3.1 8B Instruct model using transformers."""
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        llama_pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        return llama_pipeline
    
    def generate_llama_response(self, query: str, context: str, llama_pipeline) -> str:
        """Generates an answer using LLaMA 3.1 8B Instruct."""
        messages = [
            {"role": "system", "content": "You are a medical assistant. Respond to  the user's question using ONLY the provided context. If unsure, say so."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
        ]
        
        outputs = llama_pipeline(
            messages,
            max_new_tokens=512,  # Adjust based on your needs
            temperature=0.7,    # Control randomness
            top_p=0.9,           # Nucleus sampling
        )
        return outputs[0]["generated_text"][-1]["content"]

def main():
    """Main entry point of the application."""
    try:
        print("MedlinePlus Scraper and Vector Database Tool")
        print("===========================================")
        
        # Get output directory
        output_dir = input("Enter directory for MedlinePlus files (default: 'medlineplus_diseases'): ").strip()
        output_dir = output_dir if output_dir else "medlineplus_diseases"
        
        # Ask if user wants to scrape new data
        scrape_new = input("Do you want to scrape new data? (y/n, default: n): ").strip().lower()
        
        if scrape_new == 'y':
            # Initialize scraper
            scraper = MedlinePlusScraper(output_dir=output_dir)
            
            # Allow multiple letters to be scraped
            while True:
                letter = input("Enter a letter to retrieve articles (A-Z) or 'done' to continue: ").strip()
                if letter.lower() == 'done':
                    break
                
                scraper.scrape_and_save_articles(letter)
        
        # Check if output directory exists and has files
        if not os.path.exists(output_dir):
            print(f"Directory '{output_dir}' does not exist!")
            create_dir = input("Would you like to create this directory and scrape data? (y/n): ").strip().lower()
            
            if create_dir == 'y':
                os.makedirs(output_dir, exist_ok=True)
                print(f"Created directory: {output_dir}")
                
                # Initialize scraper and get data
                scraper = MedlinePlusScraper(output_dir=output_dir)
                letter = input("Enter a letter to retrieve articles (A-Z): ").strip()
                scraper.scrape_and_save_articles(letter)
            else:
                print("Exiting since no data directory exists.")
                return
        elif len(glob.glob(os.path.join(output_dir, "*.txt"))) == 0:
            print(f"No text files found in '{output_dir}'!")
            scrape_data = input("Would you like to scrape data now? (y/n): ").strip().lower()
            
            if scrape_data == 'y':
                # Initialize scraper and get data
                scraper = MedlinePlusScraper(output_dir=output_dir)
                letter = input("Enter a letter to retrieve articles (A-Z): ").strip()
                scraper.scrape_and_save_articles(letter)
            else:
                print("Exiting since no data files are available.")
                return
        
        # Get chunking parameters
        try:
            chunk_size = input("Enter chunk size in characters (default: 1000): ").strip()
            chunk_size = int(chunk_size) if chunk_size else 1000
            
            chunk_overlap = input("Enter chunk overlap in characters (default: 200): ").strip()
            chunk_overlap = int(chunk_overlap) if chunk_overlap else 200
        except ValueError:
            print("Invalid number format. Using default values.")
            chunk_size = 1000
            chunk_overlap = 200
        
        # Initialize the vectorizer
        vectorizer = MedlinePlusVectorizer(
            input_dir=output_dir,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Ask if the user wants to redo embeddings
        redo_embeddings = input("\nDo you want to redo embeddings? (y/n, default: n): ").strip().lower()
        if redo_embeddings == 'y':
            print("Redoing embeddings...")
            vectorizer.process()
        else:
            print("Skipping embeddings. Using existing vector database.")
        
        # Ask if user wants to run a test query
        run_query = input("\nWould you like to run a test query? (y/n): ").strip().lower()
        if run_query == 'y':
            query = input("Enter your query: ").strip()
            n_results = input("Enter number of results to show (default: 5): ").strip()
            n_results = int(n_results) if n_results and n_results.isdigit() else 5
            
            vectorizer.query_with_llama(query, n_results)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Program error: {e}")


if __name__ == "__main__":
    main()