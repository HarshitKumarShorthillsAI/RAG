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
from MedilinePlusVectorizer import MedlinePlusVectorizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import os
from huggingface_hub import login
import numpy as np


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


class MedlinePlusLlamaVectorizer:
    """Process MedlinePlus data into a ChromaDB vector database using Llama embeddings."""
    
    def __init__(
        self, 
        input_dir="medlineplus_diseases", 
        chunk_size=1000, 
        chunk_overlap=200,
        collection_name="medlineplus_llama_collection",
        hf_api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    ):
        """
        Initialize the Llama vectorizer.
        
        Args:
            input_dir: Directory containing scraped MedlinePlus files
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            collection_name: Name for the ChromaDB collection
            hf_api_key: Hugging Face API key for accessing Llama models
            model_name: Model name for embeddings
        """
        self.input_dir = input_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Login to Hugging Face if API key is provided
        if hf_api_key:
            login(token=hf_api_key)
        elif "HUGGINGFACE_API_KEY" in os.environ:
            print("Using Hugging Face API key from environment")
        else:
            print("WARNING: No Hugging Face API key provided. You may not be able to access Llama models.")
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize tokenizer and model for embeddings
        print(f"Loading {model_name} for embeddings...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model for embeddings
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float16,
            device_map=self.device
        )
        print(f"Model loaded on {self.device}")
        
        # Create custom embedding function for ChromaDB
        self.embedding_function = self.create_llama_embedding_function()
    
    def create_llama_embedding_function(self):
        """
        Create a custom embedding function for ChromaDB using Llama.
        
        Returns:
            Function that generates embeddings for given texts
        """
        def embed_texts(texts):
            """
            Generate embeddings for a list of texts.
            
            Args:
                texts: List of strings to embed
                
            Returns:
                List of embeddings as numpy arrays
            """
            embeddings = []
            batch_size = 8  # Adjust based on your GPU memory
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize the texts
                inputs = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512,  # Adjust based on model context window
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings - using mean pooling of last hidden states
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Use mean pooling to get sentence embeddings
                attention_mask = inputs['attention_mask']
                last_hidden_states = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
                sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
                
                # Convert to numpy and add to results
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.extend(batch_embeddings)
            
            return embeddings
        
        return embed_texts
    
    def combine_files(self):
        """
        Combine all text files in the input directory into a single string.
        
        Returns:
            Combined text from all files
        """
        # Implementation remains the same as your original
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
    
    def create_chunks(self, text):
        """
        Split the combined text into overlapping chunks.
        
        Args:
            text: The combined text to be chunked
            
        Returns:
            List of dictionaries with chunk info (id, text, metadata)
        """
        # Implementation remains the same as your original
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
    
    def create_vector_db(self, chunks):
        """
        Create a vector database from the chunks using ChromaDB with Llama embeddings.
        
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
            batch_size = 16  # Smaller batch size for Llama embeddings
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
    
    def process(self):
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
    
    def query_vector_db(self, query_text, n_results=5):
        """
        Query the vector database using Llama embeddings.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            
        Returns:
            Dictionary containing query results
        """
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            print(f"\nQuery: '{query_text}'")
            print(f"Top {n_results} results:")
            
            formatted_results = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                result_info = {
                    "rank": i+1,
                    "similarity": 1-distance,
                    "source": metadata['source'],
                    "section": metadata.get('section', 'N/A'),
                    "chunk_id": metadata['chunk_id'],
                    "text": doc
                }
                formatted_results.append(result_info)
                
                print(f"\nResult {i+1} (Similarity: {1-distance:.4f}):")
                print(f"Source: {metadata['source']}")
                if 'section' in metadata:
                    print(f"Section: {metadata['section']}")
                print(f"Text snippet: {doc[:500]}...")
                
            return {
                "query": query_text,
                "results": formatted_results
            }
                
        except Exception as e:
            print(f"Error querying the database: {e}")
            return None

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
class MedlinePlusLlamaRAG:
    """Complete RAG system for MedlinePlus using Llama for embeddings and generation."""
    
    def __init__(
        self, 
        collection_name="medlineplus_llama_collection",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        hf_api_key=os.environ.get("HUGGINGFACE_API_KEY"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=512
    ):
        """
        Initialize the complete Llama RAG system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Hugging Face model to use
            hf_api_key: Hugging Face API key for accessing Llama models
            device: Device to run the model on (cuda/cpu)
            max_new_tokens: Maximum number of tokens to generate
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        # Login to Hugging Face if API key is provided
        if hf_api_key:
            login(token=hf_api_key)
        elif "HUGGINGFACE_API_KEY" in os.environ:
            print("Using Hugging Face API key from environment")
        else:
            print("WARNING: No Hugging Face API key provided. You may not be able to access Llama models.")
        
        # Initialize vectorizer for embeddings and retrieval
        self.vectorizer = MedlinePlusLlamaVectorizer(
            collection_name=collection_name,
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Changed to use L6-v2 for embeddings
            hf_api_key=hf_api_key
        )
        
        # Initialize LLM components for generation
        print(f"Loading {model_name} for text generation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device
        )
        
        print("RAG system initialized!")
    
    def generate_answer(self, prompt):
        """
        Generate an answer using the Llama model.
        
        Args:
            prompt: Formatted prompt with context and query
            
        Returns:
            Generated answer text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the assistant's response
        # This extracts text after the last <|assistant|> tag
        assistant_part = generated_text.split("<|assistant|>")[-1].strip()
        
        return assistant_part
        
    def generate_prompt(self, query, context_docs):
        """
        Generate a prompt for the LLM using retrieved context.
        
        Args:
            query: User query string
            context_docs: List of retrieved document texts
            
        Returns:
            Formatted prompt string
        """
        # Combine context documents with clear separators
        contexts = []
        for i, doc in enumerate(context_docs):
            contexts.append(f"Document {i+1}:\n{doc}")
        
        combined_context = "\n\n".join(contexts)
        
        # Format specific to Llama 3.1 Instruct
        prompt = f"""<|begin_of_text|><|system|>
    You are a helpful, harmless, and precise medical information assistant. You will be given context information from reliable medical sources and a question. Your task is to answer the question based only on the provided context. If the context doesn't contain the answer, say you don't know based on the available information.

    Context:
    {combined_context}
    <|user|>
    {query}
    <|assistant|>
    """
        return prompt

    def answer_query(self, query, n_results=5):
        """
        End-to-end RAG pipeline to answer a user query using Llama.
        
        Args:
            query: User query string
            n_results: Number of context documents to retrieve
            
        Returns:
            Generated answer and retrieved contexts
        """
        # Step 1: Retrieve relevant documents
        print(f"Retrieving documents for: '{query}'")
        retrieval_results = self.vectorizer.query_vector_db(query, n_results)
        
        if not retrieval_results:
            return {
                "query": query,
                "answer": "Error: Failed to retrieve documents from the database.",
                "context_docs": []
            }
        
        # Extract texts from results
        retrieved_docs = [result["text"] for result in retrieval_results["results"]]
        
        # Step 2: Generate prompt with context
        prompt = self.generate_prompt(query, retrieved_docs)
        
        # Step 3: Generate answer with LLM
        print("Generating answer...")
        answer = self.generate_answer(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "context_docs": retrieved_docs,
            "retrieval_results": retrieval_results["results"]
        }

def setup_llama_rag():
    """Setup function for Llama RAG system."""
    try:
        print("\nLlama RAG System Setup")
        print("=====================")
        
        # Get Hugging Face API key
        api_key = os.environ.get("HUGGINGFACE_API_KEY")
        if not api_key:
            api_key = input("Enter your Hugging Face API key (press Enter to skip if set as env var): ").strip()
        
        # Check for existing collections - FIXED for Chroma v0.6.0
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        existing_collections = chroma_client.list_collections()  # Now returns only collection names
        
        if existing_collections:
            print("\nExisting collections:")
            for i, name in enumerate(existing_collections):
                print(f"{i+1}. {name}")
            
            collection_choice = input("\nUse existing collection? Enter number or name (or press Enter for new): ").strip()
            
            if collection_choice:
                if collection_choice.isdigit() and 1 <= int(collection_choice) <= len(existing_collections):
                    collection_name = existing_collections[int(collection_choice)-1]
                elif collection_choice in existing_collections:
                    collection_name = collection_choice
                else:
                    collection_name = input("Enter new collection name: ").strip() or "medlineplus_llama_collection"
            else:
                collection_name = input("Enter new collection name: ").strip() or "medlineplus_llama_collection"
        else:
            print("\nNo existing collections found.")
            collection_name = input("Enter collection name: ").strip() or "medlineplus_llama_collection"
        
        # Check if we need to create embeddings
        create_embeddings = False
        if collection_name not in existing_collections:
            print(f"\nCollection '{collection_name}' doesn't exist.")
            create_embeddings = input("Do you want to create embeddings now? (y/n): ").strip().lower() == 'y'
        
        if create_embeddings:
            # Get file directory
            input_dir = input("Enter directory with MedlinePlus files (default: 'medlineplus_diseases'): ").strip()
            input_dir = input_dir if input_dir else "medlineplus_diseases"
            
            # Check if directory exists and has files
            if not os.path.exists(input_dir):
                print(f"Directory '{input_dir}' does not exist!")
                return None
            
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
            
            # Create a vectorizer and process the data
            vectorizer = MedlinePlusLlamaVectorizer(
                input_dir=input_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=collection_name,
                hf_api_key=api_key,
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Changed to use L6-v2 for embeddings
            )
            
            vectorizer.process()
        
        # Specify model
        model_name = input(f"Enter model name (default: {LLM_MODEL}): ").strip()
        model_name = model_name if model_name else LLM_MODEL
        
        # Initialize the RAG system
        rag = MedlinePlusLlamaRAG(
            collection_name=collection_name,
            model_name=model_name,
            hf_api_key=api_key
        )
        
        return rag
        
    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None
def main():
    """Main entry point of the application."""
    # Add global constant for the model
    global LLM_MODEL
    LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    
    try:
        print("MedlinePlus Scraper and Vector Database Tool")
        print("===========================================")
        
        # Menu system
        print("\nPlease select an option:")
        print("1. Scrape MedlinePlus articles")
        print("2. Create standard vector database (using all-MiniLM-L6-v2)")
        print("3. Create Llama embeddings vector database")
        print("4. Run test queries on standard database")
        print("5. Run test queries on Llama database")
        print("6. Start Llama RAG system (embeddings + generation)")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            # Scraping logic
            output_dir = input("Enter directory for MedlinePlus files (default: 'medlineplus_diseases'): ").strip()
            output_dir = output_dir if output_dir else "medlineplus_diseases"
            
            scraper = MedlinePlusScraper(output_dir=output_dir)
            
            while True:
                letter = input("Enter a letter to retrieve articles (A-Z) or 'done' to continue: ").strip()
                if letter.lower() == 'done':
                    break
                
                scraper.scrape_and_save_articles(letter)
        
        elif choice == "2":
            # Standard vectorization
            output_dir = input("Enter directory with MedlinePlus files (default: 'medlineplus_diseases'): ").strip()
            output_dir = output_dir if output_dir else "medlineplus_diseases"
            
            try:
                chunk_size = input("Enter chunk size in characters (default: 1000): ").strip()
                chunk_size = int(chunk_size) if chunk_size else 1000
                
                chunk_overlap = input("Enter chunk overlap in characters (default: 200): ").strip()
                chunk_overlap = int(chunk_overlap) if chunk_overlap else 200
            except ValueError:
                print("Invalid number format. Using default values.")
                chunk_size = 1000
                chunk_overlap = 200
            
            vectorizer = MedlinePlusVectorizer(
                input_dir=output_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Explicitly set to L6-v2
            )
            
            vectorizer.process()
        
        elif choice == "3":
            # Llama vectorization
            output_dir = input("Enter directory with MedlinePlus files (default: 'medlineplus_diseases'): ").strip()
            output_dir = output_dir if output_dir else "medlineplus_diseases"
            
            # Get API key
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not api_key:
                api_key = input("Enter your Hugging Face API key (press Enter to skip if set as env var): ").strip()
            
            # Get collection name
            collection_name = input("Enter collection name (default: 'medlineplus_llama_collection'): ").strip()
            collection_name = collection_name if collection_name else "medlineplus_llama_collection"
            
            try:
                chunk_size = input("Enter chunk size in characters (default: 1000): ").strip()
                chunk_size = int(chunk_size) if chunk_size else 1000
                
                chunk_overlap = input("Enter chunk overlap in characters (default: 200): ").strip()
                chunk_overlap = int(chunk_overlap) if chunk_overlap else 200
            except ValueError:
                print("Invalid number format. Using default values.")
                chunk_size = 1000
                chunk_overlap = 200
            
            # Use MiniLM-L6-v2 for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            vectorizer = MedlinePlusLlamaVectorizer(
                input_dir=output_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=collection_name,
                hf_api_key=api_key,
                model_name=model_name
            )
            
            vectorizer.process()
        
        elif choice == "4":
            # Standard query testing
            vectorizer = MedlinePlusVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Set to L6-v2
            
            while True:
                query = input("\nEnter your query (or 'quit' to exit): ").strip()
                if query.lower() == 'quit':
                    break
                    
                n_results = input("Enter number of results to show (default: 5): ").strip()
                n_results = int(n_results) if n_results and n_results.isdigit() else 5
                
                vectorizer.query_example(query, n_results)
        
        elif choice == "5":
            # Llama query testing with MiniLM-L6-v2 for embeddings
            # Get API key
            api_key = os.environ.get("HUGGINGFACE_API_KEY")
            if not api_key:
                api_key = input("Enter your Hugging Face API key (press Enter to skip if set as env var): ").strip()
            
            # Get collection name
            collection_name = input("Enter collection name (default: 'medlineplus_llama_collection'): ").strip()
            collection_name = collection_name if collection_name else "medlineplus_llama_collection"
            
            # Set embedding model to MiniLM-L6-v2
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            vectorizer = MedlinePlusLlamaVectorizer(
                collection_name=collection_name,
                hf_api_key=api_key,
                model_name=model_name
            )
            
            while True:
                query = input("\nEnter your query (or 'quit' to exit): ").strip()
                if query.lower() == 'quit':
                    break
                    
                n_results = input("Enter number of results to show (default: 5): ").strip()
                n_results = int(n_results) if n_results and n_results.isdigit() else 5
                
                vectorizer.query_vector_db(query, n_results)
        
        elif choice == "6":
            # Llama RAG system
            llama_rag_demo()
        
        elif choice == "7":
            print("Exiting program.")
            return
        
        else:
            print("Invalid choice. Please restart the program.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Program error: {e}")

def llama_rag_demo():
    """Interactive demo for the Llama RAG system."""
    try:
        print("\nSetting up MedlinePlus Llama RAG System...")
        
        # Setup the RAG system
        rag = setup_llama_rag()
        
        if not rag:
            print("Failed to initialize RAG system.")
            return
        
        print("\nRAG system is ready! Type 'quit' to exit.")
        
        while True:
            # Get user query
            query = input("\nEnter your medical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting RAG demo.")
                break
            
            if not query:
                print("Please enter a valid question.")
                continue
            
            # Process the query
            start_time = datetime.now()
            result = rag.answer_query(query)
            end_time = datetime.now()
            
            # Display the result
            print("\n" + "="*50)
            print(f"Query: {result['query']}")
            print("="*50)
            print(f"Answer:\n{result['answer']}")
            print("-"*50)
            print(f"Processing time: {(end_time - start_time).total_seconds():.2f} seconds")
            print(f"Based on {len(result['context_docs'])} retrieved documents")
            print("="*50)
            
            # Ask if user wants to see the context documents
            show_context = input("\nWould you like to see the retrieved context? (y/n): ").strip().lower()
            if show_context == 'y':
                print("\nRetrieved Context Documents:")
                for i, doc in enumerate(result['context_docs']):
                    print(f"\n--- Document {i+1} ---")
                    print(doc[:500] + "..." if len(doc) > 500 else doc)
    
    except Exception as e:
        print(f"RAG demo error: {e}")
if __name__ == "__main__":
    main()