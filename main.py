import os
import streamlit as st
from dotenv import load_dotenv
from WebScraper import MedlinePlusScraper
from Vectorizer import MedlinePlusVectorizer
load_dotenv()

def main():
    st.title("MedlinePlus Scraper and Vector Database Tool")
    st.sidebar.title("Navigation")
    
    # Sidebar navigation
    options = ["Scrape Data", "Vectorize Data", "Query Data"]
    choice = st.sidebar.radio("Choose an option", options)
    
    if choice == "Scrape Data":
        st.header("Scrape MedlinePlus Data")
        output_dir = st.text_input("Enter directory for MedlinePlus files (default: 'medlineplus_diseases'):", "medlineplus_diseases")
        
        if st.button("Scrape New Data"):
            letter = st.text_input("Enter a letter to retrieve articles (A-Z):").strip().upper()
            
            if letter:
                with st.spinner(f"Scraping articles for letter '{letter}'..."):
                    scraper = MedlinePlusScraper(output_dir=output_dir)
                    scraper.scrape_and_save_articles(letter)
                st.success(f"Scraping completed for letter '{letter}'!")
            else:
                st.warning("Please enter a valid letter (A-Z).")
    
    elif choice == "Vectorize Data":
        st.header("Vectorize MedlinePlus Data")
        input_dir = st.text_input("Enter directory containing MedlinePlus files (default: 'medlineplus_diseases'):", "medlineplus_diseases")
        chunk_size = st.number_input("Enter chunk size in characters (default: 1000):", value=1000)
        chunk_overlap = st.number_input("Enter chunk overlap in characters (default: 200):", value=200)
        collection_name = st.text_input("Enter collection name (default: 'medlineplus_collection'):", "medlineplus_collection")
        
        if st.button("Vectorize Data"):
            vectorizer = MedlinePlusVectorizer(
                input_dir=input_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                collection_name=collection_name
            )
            
            with st.spinner("Processing data and creating vector database..."):
                vectorizer.process()
            st.success("Vectorization completed successfully!")
    
    elif choice == "Query Data":
        st.header("Query MedlinePlus Data")
        collection_name = st.text_input("Enter collection name (default: 'medlineplus_collection'):", "medlineplus_collection")
        query = st.text_input("Enter your query:")
        
        if st.button("Run Query"):
            vectorizer = MedlinePlusVectorizer(collection_name=collection_name)
            
            with st.spinner("Querying the database and generating response..."):
                # Get the answer using the RAG pipeline
                answer, context = vectorizer.query_with_rag(query)
                
                # Display the answer
                st.subheader("Generated Answer using Mistral Model")
                st.write(answer)
                
                # If context is available, display it
                if context:
                    st.subheader("Supporting Context")
                    st.write(context)
                
            st.success("Query completed!")

if __name__ == "__main__":
    main()