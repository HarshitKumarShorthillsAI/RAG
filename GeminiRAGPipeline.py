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
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiRAGPipeline:
    """
    Implements a Retrieval-Augmented Generation (RAG) pipeline using Google's Gemini model.
    
    This class extends the existing functionality to:
    1. Retrieve relevant information using vector similarity search
    2. Send query and retrieved context to Gemini model
    3. Generate contextually informed responses
    """
    
    def __init__(
        self,
        api_key: str,
        collection_name: str = "medlineplus_collection",
        model_name: str = "gemini-1.5-flash",  # Updated to recommended model
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_k: int = 5,
    ):
        """
        Initialize the Gemini RAG pipeline.
        
        Args:
            api_key: Google API key for Gemini access
            collection_name: Name of the ChromaDB collection to query
            model_name: Gemini model to use
            temperature: Controls randomness in generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_k: Number of relevant documents to retrieve
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        
        # Set up Gemini model configuration
        self.generation_config = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
            "top_p": 0.95,
            "top_k": 30
        }
        
        # Set up safety settings
        self.safety_settings = [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            }
        ]
        
        try:
            # Print available models for debugging
            print("Available Gemini models:")
            available_models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
                    print(f" - {m.name}")
            
            if self.model_name not in available_models and available_models:
                print(f"Warning: Model '{self.model_name}' not found in available models.")
                print(f"Falling back to '{available_models[0]}'")
                self.model_name = available_models[0]
            
            # Initialize the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            print(f"Initialized Gemini RAG pipeline with model: {self.model_name}")
            
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
            print("If an API key error, make sure your API key has access to the requested model.")
            raise
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: User's query text
            
        Returns:
            List of relevant documents with their metadata
        """
        try:
            collection = self.chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            # Perform vector similarity search
            results = collection.query(
                query_texts=[query],
                n_results=self.top_k
            )
            
            relevant_docs = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                relevant_docs.append({
                    "text": doc,
                    "metadata": metadata,
                    "similarity": 1 - distance  # Convert distance to similarity score
                })
            
            return relevant_docs
            
        except Exception as e:
            print(f"Error retrieving relevant documents: {e}")
            return []
    
    def format_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved documents into a context string for the model.
        
        Args:
            relevant_docs: List of relevant documents with metadata
            
        Returns:
            Formatted context string
        """
        context = "RELEVANT INFORMATION:\n\n"
        
        for i, doc in enumerate(relevant_docs):
            context += f"DOCUMENT {i+1} (Similarity: {doc['similarity']:.4f}):\n"
            context += f"Source: {doc['metadata']['source']}\n"
            if 'section' in doc['metadata']:
                context += f"Section: {doc['metadata']['section']}\n"
            context += f"Content: {doc['text']}\n\n"
        
        return context
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt to send to the Gemini model.
        
        Args:
            query: User's query
            context: Formatted context from retrieved documents
            
        Returns:
            Complete prompt for the model
        """
        prompt = f"""
You are an assistant specialized in medical information. You have access to information
from MedlinePlus, a trusted source of medical knowledge. 

Below is some relevant information retrieved based on the user's query. Use this information
to craft a comprehensive, accurate, and helpful response. If the information provided doesn't
fully address the query, acknowledge the limitations of your knowledge.

{context}

USER QUERY: {query}

Your response should:
1. Be factual and based on the retrieved information
2. Be concise but comprehensive
3. Use medical terminology appropriately, with explanations when needed
4. Cite the sources when providing specific information
5. Remind the user to consult healthcare professionals for medical advice

RESPONSE:
"""
        return prompt
    
    def query_and_generate(self, query: str) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline: retrieve, format context, generate response.
        
        Args:
            query: User's query
            
        Returns:
            Dictionary with the response and related metadata
        """
        try:
            print(f"Processing query: '{query}'")
            
            # Step 1: Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(query)
            
            if not relevant_docs:
                return {
                    "response": "I couldn't find relevant information to answer your query. Please try a different question related to medical topics available in MedlinePlus.",
                    "error": "No relevant documents found",
                    "retrieved_docs": []
                }
            
            # Step 2: Format context from retrieved documents
            context = self.format_context(relevant_docs)
            
            # Step 3: Create prompt for Gemini
            prompt = self.create_prompt(query, context)
            
            # Step 4: Generate response using Gemini
            try:
                response = self.model.generate_content(prompt)
                
                return {
                    "response": response.text,
                    "retrieved_docs": relevant_docs,
                    "prompt": prompt
                }
                
            except Exception as e:
                print(f"Error generating response with Gemini: {e}")
                return {
                    "response": "I encountered an error generating a response. Please try again or rephrase your query.",
                    "error": str(e),
                    "retrieved_docs": relevant_docs
                }
                
        except Exception as e:
            print(f"Error in RAG pipeline: {e}")
            return {
                "response": "An error occurred while processing your query. Please try again later.",
                "error": str(e),
                "retrieved_docs": []
            }
    
    def interactive_session(self):
        """
        Start an interactive RAG session with the user.
        """
        print("\n" + "="*50)
        print("Welcome to the MedlinePlus RAG with Google Gemini")
        print("Ask medical questions to get information from MedlinePlus")
        print("Type 'exit' or 'quit' to end the session")
        print("="*50 + "\n")
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Thank you for using the MedlinePlus RAG. Goodbye!")
                break
                
            if not query:
                continue
                
            # Process query through the RAG pipeline
            result = self.query_and_generate(query)
            
            print("\n" + "-"*50)
            print("RESPONSE:")
            print(result["response"])
            print("-"*50)
            
            # Option to show sources
            show_sources = input("\nWould you like to see the sources used? (y/n): ").strip().lower()
            if show_sources == 'y':
                print("\nSOURCES USED:")
                for i, doc in enumerate(result["retrieved_docs"]):
                    print(f"\nSource {i+1}: {doc['metadata']['source']}")
                    if 'section' in doc['metadata']:
                        print(f"Section: {doc['metadata']['section']}")
                    print(f"Similarity: {doc['similarity']:.4f}")

# Example usage (uncomment to run)
# if __name__ == "__main__":
#     api_key = "YOUR_API_KEY_HERE"
#     rag_pipeline = GeminiRAGPipeline(api_key=api_key)
#     rag_pipeline.interactive_session()