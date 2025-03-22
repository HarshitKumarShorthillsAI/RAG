import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import requests
import json
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO

# Import the classes to test
from WebScraper import MedlinePlusScraper
from Vectorizer import MedlinePlusVectorizer

class TestMedlinePlusScraper(unittest.TestCase):
    """Test cases for the MedlinePlusScraper class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.scraper = MedlinePlusScraper(output_dir="test_output")
        # Sample HTML content for testing
        self.sample_html = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <h1 class="with-also" itemprop="name">Test Disease</h1>
                <div class="section">
                    <div class="section-title">Causes</div>
                    <div class="section-body">This is the causes content.</div>
                </div>
                <div class="section">
                    <div class="section-title">Symptoms</div>
                    <div class="section-body">This is the symptoms content.</div>
                </div>
                <div id="mplus-content">
                    <ul>
                        <li><a href="article/disease1.htm">Disease 1</a></li>
                        <li><a href="article/disease2.htm">Disease 2</a></li>
                        <li class="special"><a href="article/special.htm">Special Link</a></li>
                    </ul>
                </div>
            </body>
        </html>
        """

    @patch('os.makedirs')
    def test_init_creates_output_directory(self, mock_makedirs):
        """Test that the constructor creates the output directory if it doesn't exist."""
        with patch('os.path.exists', return_value=False):
            scraper = MedlinePlusScraper(output_dir="new_dir")
            mock_makedirs.assert_called_once_with("new_dir")

    @patch('requests.Session.get')
    def test_retrieve_webpage_success(self, mock_get):
        """Test successful webpage retrieval."""
        mock_response = MagicMock()
        mock_response.text = self.sample_html
        mock_get.return_value = mock_response
        
        result = self.scraper.retrieve_webpage("https://example.com")
        self.assertEqual(result, self.sample_html)
        mock_get.assert_called_once_with("https://example.com", timeout=30)

    @patch('requests.Session.get')
    def test_retrieve_webpage_failure(self, mock_get):
        """Test webpage retrieval failure handling."""
        mock_get.side_effect = requests.RequestException("Connection error")
        
        result = self.scraper.retrieve_webpage("https://example.com")
        self.assertIsNone(result)

    def test_parse_article_content(self):
        """Test parsing article content from HTML."""
        result = self.scraper.parse_article_content(self.sample_html)
        
        self.assertEqual(result["Title"], "Test Disease")
        self.assertEqual(result["Causes"], "This is the causes content.")
        self.assertEqual(result["Symptoms"], "This is the symptoms content.")

    def test_parse_article_content_error_handling(self):
        """Test error handling in article content parsing."""
        result = self.scraper.parse_article_content("<invalid>html")
        self.assertIn("Error", result)

    def test_create_safe_filename(self):
        """Test creating safe filenames from article titles."""
        # Test basic sanitization
        result = self.scraper.create_safe_filename("Test Disease: A Study")
        self.assertTrue(result.startswith("Test_Disease_A_Study_"))
        
        # Test handling invalid characters
        result = self.scraper.create_safe_filename("Test/Disease\\*:?\"<>|")
        self.assertTrue(result.startswith("TestDisease_"))
        
        # Test truncation of long filenames
        long_title = "A" * 300
        result = self.scraper.create_safe_filename(long_title)
        self.assertEqual(len(result.split("_")[0]), 200)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_file(self, mock_file):
        """Test saving content to a file."""
        content = {"Title": "Test Disease", "Causes": "Test causes"}
        url = "https://example.com/test"
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 23, 12, 0, 0)
            result = self.scraper.save_to_file(content, url)
        
        self.assertFalse(result.startswith("Error"))
        mock_file.assert_called_once()
        
        # Check that the file write operations contain the expected content
        write_calls = mock_file().write.call_args_list
        self.assertTrue(any('Source: https://example.com/test' in str(call) for call in write_calls))
        self.assertTrue(any('Title' in str(call) for call in write_calls))
        self.assertTrue(any('Test Disease' in str(call) for call in write_calls))

    @patch('builtins.open')
    def test_save_to_file_handles_errors(self, mock_open):
        """Test error handling when saving to a file."""
        mock_open.side_effect = IOError("Permission denied")
        
        result = self.scraper.save_to_file({"Title": "Test"}, "https://example.com")
        self.assertTrue(result.startswith("Error"))

    @patch('requests.Session.get')
    def test_find_encyclopedia_articles(self, mock_get):
        """Test finding encyclopedia article links."""
        mock_response = MagicMock()
        mock_response.text = self.sample_html
        mock_get.return_value = mock_response
        
        result = self.scraper.find_encyclopedia_articles("A")
        
        # Should find 2 valid links (not the one with class="special")
        self.assertEqual(len(result), 2)
        self.assertTrue(all(link.startswith(self.scraper.BASE_URL) for link in result))
        self.assertTrue(any(link.endswith("disease1.htm") for link in result))
        self.assertTrue(any(link.endswith("disease2.htm") for link in result))

    def test_find_encyclopedia_articles_validates_input(self):
        """Test input validation for finding encyclopedia articles."""
        with self.assertRaises(ValueError):
            self.scraper.find_encyclopedia_articles("AB")  # Too long
        
        with self.assertRaises(ValueError):
            self.scraper.find_encyclopedia_articles("1")  # Not alphabetic
            
        with self.assertRaises(ValueError):
            self.scraper.find_encyclopedia_articles("")  # Empty

    @patch.object(MedlinePlusScraper, 'find_encyclopedia_articles')
    @patch.object(MedlinePlusScraper, 'retrieve_webpage')
    @patch.object(MedlinePlusScraper, 'parse_article_content')
    @patch.object(MedlinePlusScraper, 'save_to_file')
    def test_scrape_and_save_articles(self, mock_save, mock_parse, mock_retrieve, mock_find):
        """Test the main scraping and saving function."""
        # Setup mocks
        mock_find.return_value = ["url1", "url2"]
        mock_retrieve.return_value = self.sample_html
        mock_parse.return_value = {"Title": "Test Disease"}
        mock_save.return_value = "/path/to/saved/file.txt"
        
        # Redirect stdout to capture print statements
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            self.scraper.scrape_and_save_articles("A")
            
            # Verify function calls
            mock_find.assert_called_once_with("A")
            self.assertEqual(mock_retrieve.call_count, 2)
            self.assertEqual(mock_parse.call_count, 2)
            self.assertEqual(mock_save.call_count, 2)
            
            # Verify output
            output = captured_output.getvalue()
            self.assertIn("Found 2 articles", output)
            self.assertIn("Successfully saved 2 out of 2 articles", output)
        finally:
            sys.stdout = sys.__stdout__


class TestMedlinePlusVectorizer(unittest.TestCase):
    """Test cases for the MedlinePlusVectorizer class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Initialize with embeddings disabled for testing
        self.vectorizer = MedlinePlusVectorizer(
            input_dir="test_input",
            collection_name="test_collection",
            initialize_embeddings=False
        )
        
        # Create a sample file content
        self.sample_file_content = """
        Source: https://medlineplus.gov/ency/article/test.htm
        Extracted: 2025-03-23 12:00:00
        
        Title
        Test Disease
        
        Causes
        This is the causes section.
        
        Symptoms
        This is the symptoms section.
        """

    @patch('glob.glob')
    @patch('builtins.open', new_callable=mock_open)
    def test_combine_files(self, mock_file, mock_glob):
        """Test combining multiple files into a single text."""
        # Setup mocks
        mock_glob.return_value = ["test_input/file1.txt", "test_input/file2.txt"]
        mock_file.return_value.read.return_value = self.sample_file_content
        
        result = self.vectorizer.combine_files()
        
        # Check that files were read
        self.assertEqual(mock_file.call_count, 2)
        
        # Check that the result contains expected content
        self.assertIn("--- START OF DOCUMENT: file1.txt ---", result)
        self.assertIn("--- START OF DOCUMENT: file2.txt ---", result)
        self.assertIn("Test Disease", result)

    def test_create_chunks(self):
        """Test creating chunks from text."""
        # Use a patch to avoid initializing embeddings
        with patch.object(self.vectorizer, 'text_splitter') as mock_splitter:
            # Setup mock document return
            mock_doc = MagicMock()
            mock_doc.metadata = {}
            mock_splitter.create_documents.return_value = [mock_doc, mock_doc]
            
            result = self.vectorizer.create_chunks("Test text")
            
            # Verify the splitter was called
            mock_splitter.create_documents.assert_called_once_with(["Test text"])
            
            # Verify we got the expected documents with metadata
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].metadata["source"], "combined_text")
            self.assertIn("chunk_id", result[0].metadata)

    @patch.object(MedlinePlusVectorizer, 'combine_files')
    @patch.object(MedlinePlusVectorizer, 'create_chunks')
    @patch.object(MedlinePlusVectorizer, 'create_vector_db')
    def test_process(self, mock_create_db, mock_create_chunks, mock_combine):
        """Test the full processing pipeline."""
        # Setup mocks
        mock_combine.return_value = "Combined text"
        mock_docs = [MagicMock(), MagicMock()]
        mock_create_chunks.return_value = mock_docs
        
        self.vectorizer.process()
        
        # Verify all steps were called
        mock_combine.assert_called_once()
        mock_create_chunks.assert_called_once_with("Combined text")
        mock_create_db.assert_called_once_with(mock_docs)

    @patch('json.load')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_log_query(self, mock_exists, mock_open, mock_dump, mock_load):
        """Test logging of queries and answers."""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = [{"previous": "log"}]
        
        self.vectorizer._log_query("What are the symptoms?", "Symptoms include...")
        
        # Verify file operations
        mock_open.assert_called()
        mock_dump.assert_called_once()
        
        # Check the logged data structure
        args = mock_dump.call_args[0]
        logs = args[0]
        self.assertEqual(len(logs), 2)  # Previous log + new entry
        self.assertEqual(logs[1]["question"], "What are the symptoms?")
        self.assertEqual(logs[1]["answer"], "Symptoms include...")

    @patch('os.getenv')
    def test_initialize_mistral_model(self, mock_getenv):
        """Test initialization of the Mistral model."""
        # Setup mock
        mock_getenv.return_value = "fake_api_key"
        
        with patch('langchain_mistralai.ChatMistralAI') as mock_mistral:
            self.vectorizer.initialize_mistral_model()
            
            # Verify the model was initialized with correct parameters
            mock_mistral.assert_called_once()
            args = mock_mistral.call_args[1]
            self.assertEqual(args["model"], "mistral-large-latest")
            self.assertEqual(args["temperature"], 0.2)
            self.assertEqual(args["api_key"], "fake_api_key")

    @patch('os.getenv')
    def test_initialize_mistral_model_missing_api_key(self, mock_getenv):
        """Test error handling when API key is missing."""
        # Setup mock to return None (missing API key)
        mock_getenv.return_value = None
        
        with self.assertRaises(ValueError):
            self.vectorizer.initialize_mistral_model()

    @patch.object(MedlinePlusVectorizer, 'initialize_mistral_model')
    @patch.object(MedlinePlusVectorizer, '_log_query')
    def test_query_with_rag(self, mock_log, mock_init_model):
        """Test querying with the RAG pipeline."""
        # Setup mocks
        mock_model = MagicMock()
        mock_init_model.return_value = mock_model
        
        with patch.object(self.vectorizer, 'initialize_rag_pipeline') as mock_init_rag:
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = "This is the answer"
            mock_init_rag.return_value = mock_pipeline
            
            result, context = self.vectorizer.query_with_rag("What is this disease?")
            
            # Verify the RAG pipeline was initialized and used
            mock_init_rag.assert_called_once()
            mock_pipeline.run.assert_called_once_with("What is this disease?")
            
            # Verify the result
            self.assertEqual(result, "This is the answer")
            self.assertEqual(context, "")  # Context should be empty
            
            # Verify logging
            mock_log.assert_called_once_with("What is this disease?", "This is the answer")


if __name__ == '__main__':
    unittest.main()