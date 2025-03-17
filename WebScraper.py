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
