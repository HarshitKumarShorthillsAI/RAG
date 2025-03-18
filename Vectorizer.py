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
from langchain_mistralai import ChatMistralAI
import getpass
from dotenv import load_dotenv
import os
import streamlit as st
import torch
load_dotenv()

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
    
    def query_with_mistral(self, query_text: str, n_results: int = 5) -> None:
        """Query the ChromaDB vector database and generate an answer using Mistral."""
        try:
            # Debug: Print API key
            load_dotenv()
            # st.write("Mistral API Key:", os.getenv("MISTRAL_API_KEY"))

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
            
            # Initialize the Mistral model
            mistral_llm = self.initialize_mistral_model()
            
            # Generate an answer using Mistral
            answer = self.generate_mistral_response(query_text, context, mistral_llm)
            
            # Display the results in the Streamlit UI
            st.subheader("Generated Answer using Mistral Model")
            st.write(answer)
            
            st.subheader("Supporting Context")
            st.write(context[:1000] + "...")  # Show first 1000 chars of context
            
        except Exception as e:
            st.error(f"Error querying the database or generating response: {e}")
    def initialize_mistral_model(self):
        """Initializes the Mistral model using LangChain."""
        # Prompt the user for the Mistral API key if not already set
        load_dotenv()
        mistral_api_key =  os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            mistral_api_key = getpass.getpass("Enter your Mistral API key: ")
            if not mistral_api_key:
                raise ValueError("Mistral API key is required. Please provide a valid API key.")
        
        # Initialize the Mistral model with the API key
        llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0.2,  # Lower temperature for more factual responses
            max_retries=2,     # Retry on API failures
            api_key=mistral_api_key  # Pass the API key directly
        )
        return llm
    
    def generate_mistral_response(self,query, context, mistral_llm):
        """Generates an answer using Mistral."""
        prompt = (
            "You are a medical assistant. Answer the user's question using ONLY the provided context. "
            "If unsure, say so. Always explain medical terms in simple language.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}"
        )
    
        response = mistral_llm.invoke(prompt)
        return response.content