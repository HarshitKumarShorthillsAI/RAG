from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain.schema import Document
import chromadb
import os
import glob
import uuid
import tqdm
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()

class MedlinePlusVectorizer:
    """Process MedlinePlus data into a ChromaDB vector database and implement RAG pipeline."""
    
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
        
        # Initialize ChromaDB client for document processing
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use HuggingFaceEmbeddings for LangChain compatibility
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
        # Get the Mistral API key from environment variables
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
            with open(log_file, "r") as f:
                logs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)