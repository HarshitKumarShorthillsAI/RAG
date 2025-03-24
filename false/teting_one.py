import pytest
import os
import json
import shutil
import tempfile
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
from bs4 import BeautifulSoup
import requests
from datetime import datetime

# Import the classes to test
from WebScraper import MedlinePlusScraper
from Vectorizer import MedlinePlusVectorizer

# Test data
MOCK_HTML = """
<html>
    <body>
        <h1 class="with-also" itemprop="name">Test Disease</h1>
        <div class="section">
            <div class="section-title">Definition</div>
            <div class="section-body">This is a test disease definition.</div>
        </div>
        <div class="section">
            <div class="section-title">Causes</div>
            <div class="section-body">These are the causes of the test disease.</div>
        </div>
        <div id="mplus-content">
            <ul>
                <li><a href="article/disease1.htm">Disease 1</a></li>
                <li><a href="article/disease2.htm">Disease 2</a></li>
            </ul>
        </div>
    </body>
</html>
"""

MOCK_ARTICLE_LINKS = [
    "https://medlineplus.gov/ency/article/disease1.htm",
    "https://medlineplus.gov/ency/article/disease2.htm"
]

MOCK_COMBINED_TEXT = """
--- START OF DOCUMENT: file1.txt ---
Title: Disease 1
Definition: This is disease 1.
--- END OF DOCUMENT: file1.txt ---

--- START OF DOCUMENT: file2.txt ---
Title: Disease 2
Definition: This is disease 2.
--- END OF DOCUMENT: file2.txt ---
"""

class TestReport:
    """Class to manage test results and generate Excel report."""
    
    def __init__(self, report_path="test_results.xlsx"):
        self.report_path = report_path
        self.results = []
        
    def add_result(self, test_case_id, section, subsection, title, description, 
                  preconditions, test_data, test_steps, expected_result, status, actual_result=""):
        """Add a test result to the collection."""
        self.results.append({
            "TEST CASE ID": test_case_id,
            "SECTION": section,
            "SUB-SECTION": subsection,
            "TEST CASE TITLE": title,
            "TEST DESCRIPTION": description,
            "PRECONDITIONS": preconditions,
            "TEST DATA": test_data,
            "TEST STEPS": test_steps,
            "EXPECTED RESULT": expected_result,
            "ACTUAL RESULT": actual_result,
            "STATUS": status
        })
        
    def generate_report(self):
        """Generate an Excel report of test results."""
        df = pd.DataFrame(self.results)
        df.to_excel(self.report_path, index=False)
        print(f"Test report generated at {self.report_path}")


# Initialize the test report
test_report = TestReport()


# Fixture for temporary directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# MedlinePlusScraper Tests
class TestMedlinePlusScraper:
    
    @pytest.fixture
    def scraper(self, temp_dir):
        """Create a scraper instance with a temporary output directory."""
        return MedlinePlusScraper(output_dir=temp_dir)
    
    def test_retrieve_webpage_success(self, scraper):
        """Test TC001: Test that webpage retrieval works with a valid URL."""
        test_id = "TC001"
        section = "MedlinePlusScraper"
        subsection = "retrieve_webpage"
        
        # Setup test
        with patch.object(scraper.session, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.text = MOCK_HTML
            mock_get.return_value = mock_response
            
            # Execute test
            result = scraper.retrieve_webpage("https://example.com")
            
            # Verify results
            status = "PASS" if result == MOCK_HTML else "FAIL"
            actual_result = f"Retrieved {len(result)} characters" if result else "Failed to retrieve content"
            
            # Record test result
            test_report.add_result(
                test_id, section, subsection,
                "Webpage Retrieval Success",
                "Verify that the retrieve_webpage method successfully retrieves content from a valid URL",
                "Scraper instance initialized",
                "Mock URL: https://example.com",
                "1. Call retrieve_webpage with valid URL\n2. Check if content is returned",
                "HTML content should be returned",
                status, actual_result
            )
            
            assert result == MOCK_HTML
    
    def test_retrieve_webpage_failure(self, scraper):
        """Test TC002: Test that webpage retrieval handles errors gracefully."""
        test_id = "TC002"
        section = "MedlinePlusScraper"
        subsection = "retrieve_webpage"
        
        # Setup test
        with patch.object(scraper.session, 'get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection error")
            
            # Execute test
            result = scraper.retrieve_webpage("https://invalid-url.com")
            
            # Verify results
            status = "PASS" if result is None else "FAIL"
            actual_result = "None returned as expected" if result is None else f"Unexpected result: {result}"
            
            # Record test result
            test_report.add_result(
                test_id, section, subsection,
                "Webpage Retrieval Failure",
                "Verify that the retrieve_webpage method handles connection errors gracefully",
                "Scraper instance initialized",
                "Mock URL that throws exception: https://invalid-url.com",
                "1. Call retrieve_webpage with URL that causes exception\n2. Check if None is returned",
                "None should be returned, no exception raised",
                status, actual_result
            )
            
            assert result is None
    
    def test_parse_article_content(self, scraper):
        """Test TC003: Test article content parsing from HTML."""
        test_id = "TC003"
        section = "MedlinePlusScraper"
        subsection = "parse_article_content"
        
        # Execute test
        result = scraper.parse_article_content(MOCK_HTML)
        
        # Verify results
        expected_keys = ["Title", "Definition", "Causes"]
        all_keys_present = all(key in result for key in expected_keys)
        status = "PASS" if all_keys_present and result["Title"] == "Test Disease" else "FAIL"
        actual_result = f"Parsed keys: {list(result.keys())}" if result else "Failed to parse content"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Article Content Parsing",
            "Verify that article content is correctly parsed from HTML",
            "Scraper instance initialized",
            "Mock HTML with article content structure",
            "1. Call parse_article_content with HTML content\n2. Check if all expected sections are extracted",
            "Dictionary with Title, Definition and Causes keys should be returned",
            status, actual_result
        )
        
        assert "Title" in result
        assert result["Title"] == "Test Disease"
        assert "Definition" in result
        assert "Causes" in result
    
    def test_find_encyclopedia_articles(self, scraper):
        """Test TC004: Test finding article links for a given letter."""
        test_id = "TC004"
        section = "MedlinePlusScraper"
        subsection = "find_encyclopedia_articles"
        
        # Setup test
        with patch.object(scraper, 'retrieve_webpage') as mock_retrieve:
            mock_retrieve.return_value = MOCK_HTML
            
            # Execute test
            result = scraper.find_encyclopedia_articles("A")
            
            # Verify results
            expected_urls = [
                "https://medlineplus.gov/ency/article/disease1.htm",
                "https://medlineplus.gov/ency/article/disease2.htm"
            ]
            urls_match = sorted(result) == sorted(expected_urls)
            status = "PASS" if urls_match else "FAIL"
            actual_result = f"Found {len(result)} article links" if result else "No article links found"
            
            # Record test result
            test_report.add_result(
                test_id, section, subsection,
                "Find Encyclopedia Articles",
                "Verify that article links are correctly extracted for a given letter",
                "Scraper instance initialized",
                "Letter: A, Mock HTML with article links",
                "1. Call find_encyclopedia_articles with letter 'A'\n2. Check if article links are extracted",
                "List of article URLs should be returned",
                status, actual_result
            )
            
            assert sorted(result) == sorted(expected_urls)
    
    def test_create_safe_filename(self, scraper):
        """Test TC005: Test creation of safe filenames."""
        test_id = "TC005"
        section = "MedlinePlusScraper"
        subsection = "create_safe_filename"
        
        # Execute test
        unsafe_title = "Disease: With * Invalid / Chars?"
        result = scraper.create_safe_filename(unsafe_title)
        
        # Verify results
        has_no_invalid_chars = all(c not in result for c in r'\\/*?:"<>|')
        status = "PASS" if has_no_invalid_chars and "Disease_With_Invalid_Chars" in result else "FAIL"
        actual_result = f"Created filename: {result}"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Create Safe Filename",
            "Verify that invalid characters are removed from filenames",
            "Scraper instance initialized",
            "Unsafe title: 'Disease: With * Invalid / Chars?'",
            "1. Call create_safe_filename with unsafe title\n2. Check if result contains no invalid characters",
            "Filename without invalid characters should be returned",
            status, actual_result
        )
        
        assert all(c not in result for c in r'\\/*?:"<>|')
        assert "Disease_With_Invalid_Chars" in result
    
    def test_save_to_file(self, scraper, temp_dir):
        """Test TC006: Test saving content to a file."""
        test_id = "TC006"
        section = "MedlinePlusScraper"
        subsection = "save_to_file"
        
        # Setup test
        content = {
            "Title": "Test Disease",
            "Definition": "This is a test definition."
        }
        url = "https://example.com/test"
        
        # Execute test
        filepath = scraper.save_to_file(content, url)
        file_exists = os.path.exists(filepath)
        
        # Verify file content
        if file_exists:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
        
        # Verify results
        has_expected_content = file_exists and "Test Disease" in file_content and "This is a test definition." in file_content
        status = "PASS" if has_expected_content else "FAIL"
        actual_result = f"File saved at: {filepath}" if file_exists else "Failed to save file"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Save Content to File",
            "Verify that content is correctly saved to a file",
            "Scraper instance initialized with temporary directory",
            "Test content dictionary with Title and Definition",
            "1. Call save_to_file with content and URL\n2. Check if file exists and contains expected content",
            "File should be created with proper content",
            status, actual_result
        )
        
        assert file_exists
        assert "Test Disease" in file_content
        assert "This is a test definition." in file_content


# MedlinePlusVectorizer Tests
class TestMedlinePlusVectorizer:
    
    @pytest.fixture
    def vectorizer(self, temp_dir):
        """Create a vectorizer instance with a temporary directory."""
        # Create test input directory
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        # Create test files
        with open(os.path.join(input_dir, "file1.txt"), 'w') as f:
            f.write("Title: Disease 1\nDefinition: This is disease 1.")
        with open(os.path.join(input_dir, "file2.txt"), 'w') as f:
            f.write("Title: Disease 2\nDefinition: This is disease 2.")
        
        return MedlinePlusVectorizer(input_dir=input_dir, collection_name="test_collection", initialize_embeddings=False)
    
    def test_combine_files(self, vectorizer):
        """Test TC007: Test combining files from the input directory."""
        test_id = "TC007"
        section = "MedlinePlusVectorizer"
        subsection = "combine_files"
        
        # Execute test
        result = vectorizer.combine_files()
        
        # Verify results
        contains_file1 = "Title: Disease 1" in result
        contains_file2 = "Title: Disease 2" in result
        status = "PASS" if contains_file1 and contains_file2 else "FAIL"
        actual_result = f"Combined text length: {len(result)} characters"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Combine Files",
            "Verify that files are correctly combined from input directory",
            "Vectorizer instance initialized with test files",
            "Two test files with disease information",
            "1. Call combine_files method\n2. Check if result contains content from both files",
            "Combined text should contain content from all files",
            status, actual_result
        )
        
        assert "Title: Disease 1" in result
        assert "Title: Disease 2" in result
    
    def test_create_chunks(self, vectorizer):
        """Test TC008: Test creating chunks from combined text."""
        test_id = "TC008"
        section = "MedlinePlusVectorizer"
        subsection = "create_chunks"
        
        # Execute test
        chunks = vectorizer.create_chunks(MOCK_COMBINED_TEXT)
        
        # Verify results
        all_chunks_have_metadata = all("metadata" in chunk and "chunk_id" in chunk.metadata for chunk in chunks)
        status = "PASS" if len(chunks) > 0 and all_chunks_have_metadata else "FAIL"
        actual_result = f"Created {len(chunks)} chunks with metadata"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Create Chunks",
            "Verify that text is correctly chunked with proper metadata",
            "Vectorizer instance initialized",
            "Mock combined text with START/END document markers",
            "1. Call create_chunks method with combined text\n2. Check if chunks are created with metadata",
            "List of Document objects with metadata should be returned",
            status, actual_result
        )
        
        assert len(chunks) > 0
        assert all("metadata" in chunk for chunk in chunks)
        assert all("chunk_id" in chunk.metadata for chunk in chunks)
    
    @patch("langchain.vectorstores.Chroma.from_documents")
    def test_create_vector_db(self, mock_from_documents, vectorizer):
        """Test TC009: Test creating a vector database from documents."""
        test_id = "TC009"
        section = "MedlinePlusVectorizer"
        subsection = "create_vector_db"
        
        # Setup mock
        mock_vector_store = MagicMock()
        mock_from_documents.return_value = mock_vector_store
        
        # Create test documents
        from langchain.schema import Document
        docs = [
            Document(page_content="Test content 1", metadata={"chunk_id": "1"}),
            Document(page_content="Test content 2", metadata={"chunk_id": "2"})
        ]
        
        # Execute test
        vectorizer.create_vector_db(docs)
        
        # Verify results
        mock_called = mock_from_documents.called
        status = "PASS" if mock_called else "FAIL"
        actual_result = "Vector store creation method called successfully" if mock_called else "Vector store creation method not called"
        
        # Record test result
        test_report.add_result(
            test_id, section, subsection,
            "Create Vector Database",
            "Verify that vector database is created from documents",
            "Vectorizer instance initialized",
            "Two test Document objects",
            "1. Call create_vector_db method with documents\n2. Check if Chroma.from_documents was called",
            "Chroma.from_documents should be called to create vector store",
            status, actual_result
        )
        
        assert mock_from_documents.called
        mock_vector_store.persist.assert_called_once()
    
    @patch("chromadb.PersistentClient")
    @patch("langchain.vectorstores.Chroma")
    def test_initialize_rag_pipeline(self, mock_chroma, mock_client, vectorizer):
        """Test TC010: Test initializing the RAG pipeline."""
        test_id = "TC010"
        section = "MedlinePlusVectorizer"
        subsection = "initialize_rag_pipeline"
        
        # Setup mocks
        mock_vector_store = MagicMock()
        mock_chroma.return_value = mock_vector_store
        mock_vector_store.as_retriever.return_value = MagicMock()
        
        with patch.object(vectorizer, 'initialize_mistral_model') as mock_init_model:
            mock_init_model.return_value = MagicMock()
            
            # Mock RetrievalQA
            with patch('langchain.chains.RetrievalQA.from_chain_type') as mock_retrieval_qa:
                mock_retrieval_qa.return_value = MagicMock()
                
                # Execute test
                try:
                    result = vectorizer.initialize_rag_pipeline()
                    exception_raised = False
                except Exception as e:
                    exception_raised = True
                    error_message = str(e)
                
                # Verify results
                status = "PASS" if not exception_raised and mock_retrieval_qa.called else "FAIL"
                actual_result = "RAG pipeline initialized successfully" if not exception_raised else f"Exception: {error_message}"
                
                # Record test result
                test_report.add_result(
                    test_id, section, subsection,
                    "Initialize RAG Pipeline",
                    "Verify that RAG pipeline is correctly initialized",
                    "Vectorizer instance initialized",
                    "Mocked vector store and LLM model",
                    "1. Call initialize_rag_pipeline method\n2. Check if RetrievalQA.from_chain_type was called",
                    "RetrievalQA chain should be created and returned",
                    status, actual_result
                )
                
                assert not exception_raised
                assert mock_retrieval_qa.called
    
    @patch('os.getenv')
    def test_initialize_mistral_model(self, mock_getenv, vectorizer):
        """Test TC011: Test initializing the Mistral model."""
        test_id = "TC011"
        section = "MedlinePlusVectorizer"
        subsection = "initialize_mistral_model"
        
        # Setup mock
        mock_getenv.return_value = "fake_api_key"
        
        with patch('langchain_mistralai.ChatMistralAI') as mock_mistral:
            mock_mistral.return_value = MagicMock()
            
            # Execute test
            try:
                result = vectorizer.initialize_mistral_model()
                exception_raised = False
            except Exception as e:
                exception_raised = True
                error_message = str(e)
            
            # Verify results
            status = "PASS" if not exception_raised and mock_mistral.called else "FAIL"
            actual_result = "Mistral model initialized successfully" if not exception_raised else f"Exception: {error_message}"
            
            # Record test result
            test_report.add_result(
                test_id, section, subsection,
                "Initialize Mistral Model",
                "Verify that Mistral model is correctly initialized",
                "Vectorizer instance initialized",
                "Mocked API key: fake_api_key",
                "1. Call initialize_mistral_model method\n2. Check if ChatMistralAI constructor was called with correct parameters",
                "ChatMistralAI instance should be created and returned",
                status, actual_result
            )
            
            assert not exception_raised
            assert mock_mistral.called
            # Check if the model was initialized with correct parameters
            mock_mistral.assert_called_with(
                model="mistral-large-latest",
                temperature=0.2,
                max_retries=2,
                api_key="fake_api_key"
            )
    
    def test_log_query(self, vectorizer, temp_dir):
        """Test TC012: Test logging queries and answers."""
        test_id = "TC012"
        section = "MedlinePlusVectorizer"
        subsection = "_log_query"
        
        # Setup - redirect log file to temp dir
        log_file_path = os.path.join(temp_dir, "query_logs.json")
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', new_callable=mock_open) as mock_file, \
             patch('json.load') as mock_load, \
             patch('json.dump') as mock_dump:
            
            mock_exists.return_value = False
            mock_load.return_value = []
            
            # Execute test
            vectorizer._log_query("What is diabetes?", "Diabetes is a chronic condition...")
            
            # Verify results
            file_opened = mock_file.called
            json_dumped = mock_dump.called
            status = "PASS" if file_opened and json_dumped else "FAIL"
            actual_result = "Query logged successfully" if file_opened and json_dumped else "Failed to log query"
            
            # Record test result
            test_report.add_result(
                test_id, section, subsection,
                "Log Query",
                "Verify that queries and answers are correctly logged",
                "Vectorizer instance initialized",
                "Test query: 'What is diabetes?', Test answer: 'Diabetes is a chronic condition...'",
                "1. Call _log_query method with query and answer\n2. Check if data is written to JSON file",
                "Log entry should be added to the JSON log file",
                status, actual_result
            )
            
            assert file_opened
            assert json_dumped
            # Check if json.dump was called with a list containing our entry
            args, _ = mock_dump.call_args
            assert len(args) >= 1
            log_entries = args[0]
            assert len(log_entries) == 1
            assert log_entries[0]["question"] == "What is diabetes?"
            assert log_entries[0]["answer"] == "Diabetes is a chronic condition..."


def main():
    """Run the tests and generate the report."""
    # Run pytest programmatically
    pytest.main(["-v", __file__])
    
    # Generate the Excel report
    test_report.generate_report()


if __name__ == "__main__":
    main()