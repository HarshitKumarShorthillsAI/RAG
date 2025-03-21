import os
import sys
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import openpyxl
from datetime import datetime

# Import the classes to test
sys.path.append(".")
from WebScraper import MedlinePlusScraper
from Vectorizer import MedlinePlusVectorizer
from langchain.schema import Document

# ============ MedlinePlusScraper Tests ============

class TestMedlinePlusScraper:
    """Test suite for MedlinePlusScraper class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, temp_dir):
        """Test if scraper initializes correctly and creates output dir."""
        scraper = MedlinePlusScraper(output_dir=temp_dir)
        assert os.path.exists(temp_dir)
        
        new_dir = os.path.join(temp_dir, "new_dir")
        scraper = MedlinePlusScraper(output_dir=new_dir)
        assert os.path.exists(new_dir)
    
    def test_parse_article_content(self):
        """Test parsing of article content from HTML."""
        html = """
        <html>
            <body>
                <h1 class="with-also" itemprop="name">Test Disease</h1>
                <div class="section">
                    <div class="section-title">Symptoms</div>
                    <div class="section-body">Test symptoms description</div>
                </div>
                <div class="section">
                    <div class="section-title">Causes</div>
                    <div class="section-body">Test causes description</div>
                </div>
            </body>
        </html>
        """
        scraper = MedlinePlusScraper()
        result = scraper.parse_article_content(html)
        assert result["Title"] == "Test Disease"
        assert result["Symptoms"] == "Test symptoms description"
        assert result["Causes"] == "Test causes description"
    
    def test_create_safe_filename(self):
        """Test conversion of article titles to safe filenames."""
        scraper = MedlinePlusScraper()
        result = scraper.create_safe_filename("Arthritis, Rheumatoid: A Guide/Overview")
        assert ":" not in result
        assert "/" not in result
        assert "," not in result
        assert result.startswith("Arthritis_Rheumatoid")
        
        long_title = "A" * 300
        result = scraper.create_safe_filename(long_title)
        assert len(result) < 260  # Windows path limit
    
    @patch('builtins.open', new_callable=MagicMock)
    def test_save_to_file(self, mock_open, temp_dir):
        """Test saving content to file."""
        scraper = MedlinePlusScraper(output_dir=temp_dir)
        content = {
            "Title": "Test Disease",
            "Symptoms": "Test symptoms",
            "Causes": "Test causes"
        }
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        result = scraper.save_to_file(content, "https://example.com")
        assert mock_open.called
        assert mock_file.write.called
        
        mock_open.side_effect = IOError("Permission denied")
        result = scraper.save_to_file(content, "https://example.com")
        assert result.startswith("Error")
    
    @patch('WebScraper.MedlinePlusScraper.retrieve_webpage')
    def test_find_encyclopedia_articles(self, mock_retrieve):
        """Test finding article links for a letter."""
        html = """
        <html>
            <body>
                <div id="mplus-content">
                    <ul>
                        <li><a href="article/000001.htm">Article 1</a></li>
                        <li><a href="article/000002.htm">Article 2</a></li>
                        <li class="nolink">Not a link</li>
                        <li><a href="not_article/000003.htm">Not an article</a></li>
                    </ul>
                </div>
            </body>
        </html>
        """
        mock_retrieve.return_value = html
        scraper = MedlinePlusScraper()
        result = scraper.find_encyclopedia_articles("A")
        assert len(result) == 2
        assert all(link.startswith(scraper.BASE_URL + "article/") for link in result)
        
        with pytest.raises(ValueError):
            scraper.find_encyclopedia_articles("123")
        
        mock_retrieve.return_value = None
        result = scraper.find_encyclopedia_articles("A")
        assert result == []

    @patch('builtins.open', new_callable=MagicMock)
    def test_save_to_file_missing_keys(self, mock_open, temp_dir):
        """Test saving content with missing keys."""
        scraper = MedlinePlusScraper(output_dir=temp_dir)
        content = {
            "Title": "Test Disease",
        }
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        result = scraper.save_to_file(content, "https://example.com")
        assert mock_open.called
        assert mock_file.write.called
        assert "Symptoms" not in content

# ============ MedlinePlusVectorizer Tests ============

class TestMedlinePlusVectorizer:
    """Test suite for MedlinePlusVectorizer class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        for i in range(3):
            with open(os.path.join(temp_dir, f"test_{i}.txt"), "w") as f:
                f.write(f"Test content {i}")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_chroma_client(self):
        """Create a mock ChromaDB client."""
        with patch('chromadb.PersistentClient') as mock:
            yield mock
    
    def test_initialization(self, mock_chroma_client):
        """Test if vectorizer initializes correctly."""
        vectorizer = MedlinePlusVectorizer(
            input_dir="test_dir",
            chunk_size=500,
            chunk_overlap=100,
            collection_name="test_collection"
        )
        assert vectorizer.input_dir == "test_dir"
        assert vectorizer.chunk_size == 500
        assert vectorizer.chunk_overlap == 100
        assert vectorizer.collection_name == "test_collection"
        assert mock_chroma_client.called
    
    def test_combine_files(self, temp_dir):
        """Test combining files into a single string."""
        vectorizer = MedlinePlusVectorizer(input_dir=temp_dir)
        result = vectorizer.combine_files()
        assert "Test content 0" in result
        assert "Test content 1" in result
        assert "Test content 2" in result
        assert "START OF DOCUMENT: test_0.txt" in result
        assert "END OF DOCUMENT: test_2.txt" in result

    def test_combine_files_empty_dir(self, temp_dir):
        """Test combining files from an empty directory."""
        empty_dir = os.path.join(temp_dir, "empty_dir")
        os.makedirs(empty_dir)
        vectorizer = MedlinePlusVectorizer(input_dir=empty_dir)
        result = vectorizer.combine_files()
        assert result == ""
    
    def test_create_chunks(self):
        """Test creating document chunks from text."""
        vectorizer = MedlinePlusVectorizer(chunk_size=10, chunk_overlap=2)
        text = "This is a test text for chunking. It should be split into multiple chunks."
        documents = vectorizer.create_chunks(text)
        assert len(documents) > 1
        for doc in documents:
            assert "source" in doc.metadata
            assert "chunk_id" in doc.metadata

    def test_create_chunks_small_chunk_size(self):
        """Test creating document chunks with a very small chunk size."""
        vectorizer = MedlinePlusVectorizer(chunk_size=5, chunk_overlap=1)
        text = "This is a test text for chunking."
        documents = vectorizer.create_chunks(text)
        assert len(documents) > 1
        for doc in documents:
            assert len(doc.page_content) <= 5
    
    @patch('langchain.vectorstores.Chroma.from_documents')
    def test_create_vector_db(self, mock_from_documents):
        """Test creating vector database from documents."""
        vectorizer = MedlinePlusVectorizer()
        documents = [
            Document(page_content="Test content 1", metadata={"source": "test", "chunk_id": "1"}),
            Document(page_content="Test content 2", metadata={"source": "test", "chunk_id": "2"})
        ]
        mock_vector_store = MagicMock()
        mock_from_documents.return_value = mock_vector_store
        vectorizer.create_vector_db(documents)
        assert mock_from_documents.called
        assert mock_vector_store.persist.called

    @patch('langchain.vectorstores.Chroma.from_documents')
    def test_create_vector_db_empty_documents(self, mock_from_documents):
        """Test creating vector database with an empty documents list."""
        vectorizer = MedlinePlusVectorizer()
        documents = []
        mock_vector_store = MagicMock()
        mock_from_documents.return_value = mock_vector_store
        vectorizer.create_vector_db(documents)
        assert mock_from_documents.called
        assert mock_vector_store.persist.called
    
    @patch('Vectorizer.MedlinePlusVectorizer.combine_files')
    @patch('Vectorizer.MedlinePlusVectorizer.create_chunks')
    @patch('Vectorizer.MedlinePlusVectorizer.create_vector_db')
    def test_process(self, mock_create_vector_db, mock_create_chunks, mock_combine_files):
        """Test the entire processing pipeline."""
        mock_combine_files.return_value = "Combined text"
        mock_create_chunks.return_value = [Document(page_content="Chunk")]
        vectorizer = MedlinePlusVectorizer()
        vectorizer.process()
        assert mock_combine_files.called
        mock_create_chunks.assert_called_once_with("Combined text")
        assert mock_create_vector_db.called
    
    @patch('langchain.embeddings.HuggingFaceEmbeddings', autospec=True)
    @patch('os.getenv')
    def test_initialize_mistral_model(self, mock_getenv, mock_embeddings):
        """Test initialization of Mistral model."""
        mock_getenv.return_value = "test_api_key"
        vectorizer = MedlinePlusVectorizer()
        result = vectorizer.initialize_mistral_model()
        mock_getenv.assert_called_with("MISTRAL_API_KEY")
        mock_getenv.reset_mock()
        mock_getenv.return_value = None
        with pytest.raises(ValueError):
            vectorizer.initialize_mistral_model()

    @patch('langchain.embeddings.HuggingFaceEmbeddings', autospec=True)
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    @patch('json.dump')
    def test_log_query(self, mock_dump, mock_load, mock_open, mock_embeddings):
        """Test logging of queries and answers."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_load.return_value = []
        vectorizer = MedlinePlusVectorizer(initialize_embeddings=False)
        vectorizer._log_query("Test query", "Test answer")
        assert mock_open.called
        assert mock_load.called
        assert mock_dump.called
        mock_load.return_value = [{"existing": "log"}]
        vectorizer._log_query("Test query", "Test answer")
        mock_dump.assert_called_with([{"existing": "log"}, mock_dump.call_args[0][0][1]], mock_file, indent=4)

# ============ Excel Reporting ============

def write_test_results_to_excel(results, filename="test_results.xlsx"):
    """Write test results to an Excel file."""
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Test Results"
    
    # Write headers
    sheet.append(["Test Case", "Description", "Expected Outcome", "Status", "Timestamp"])
    
    # Write test results
    for test_case, data in results.items():
        sheet.append([
            test_case,
            data["description"],
            data["expected_outcome"],
            data["status"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])
    
    # Save the workbook
    workbook.save(filename)
    print(f"Test results saved to {filename}")

# ============ Main Execution ============

if __name__ == "__main__":
    # Run pytest and capture results
    result = pytest.main(["-v"])
    
    # Collect test results
    test_results = {
        "test_initialization": {
            "description": "Tests if the scraper initializes correctly and creates the output directory.",
            "expected_outcome": "The scraper should create the output directory if it doesn’t exist and initialize without errors.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_parse_article_content": {
            "description": "Tests the parsing of article content from HTML.",
            "expected_outcome": "The parsed content should match the expected dictionary structure with the correct values.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_create_safe_filename": {
            "description": "Tests the conversion of article titles to safe filenames.",
            "expected_outcome": "The filename should not contain special characters and should be truncated if too long.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_save_to_file": {
            "description": "Tests saving parsed content to a file and handles file operation errors gracefully.",
            "expected_outcome": "The content should be saved to a file, and errors (e.g., permission denied) should be handled without crashing.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_find_encyclopedia_articles": {
            "description": "Tests the extraction of valid article links from a webpage for a given letter.",
            "expected_outcome": "The scraper should return a list of valid article links and handle invalid inputs (e.g., non-letter characters) gracefully.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_save_to_file_missing_keys": {
            "description": "Tests saving content with missing keys (e.g., missing 'Symptoms' or 'Causes').",
            "expected_outcome": "The scraper should handle missing keys gracefully and still save the available content.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_initialization": {
            "description": "Tests if the vectorizer initializes correctly with the provided parameters.",
            "expected_outcome": "The vectorizer should initialize with the correct input directory, chunk size, chunk overlap, and collection name.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_combine_files": {
            "description": "Tests combining multiple files into a single string with appropriate separators.",
            "expected_outcome": "The combined text should include the content of all files and proper document separators.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_combine_files_empty_dir": {
            "description": "Tests combining files from an empty directory.",
            "expected_outcome": "The function should return an empty string for an empty directory.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_create_chunks": {
            "description": "Tests splitting text into chunks of the specified size and overlap.",
            "expected_outcome": "The text should be split into multiple chunks, each with the correct metadata.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_create_chunks_small_chunk_size": {
            "description": "Tests chunk creation with a very small chunk size.",
            "expected_outcome": "The text should be split into multiple small chunks, each respecting the chunk size limit.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_create_vector_db": {
            "description": "Tests creating a vector database from a list of documents.",
            "expected_outcome": "The vector database should be created, and the 'persist' method should be called.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_create_vector_db_empty_documents": {
            "description": "Tests creating a vector database with an empty list of documents.",
            "expected_outcome": "The vector database should be created without errors, even with an empty document list.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_process": {
            "description": "Tests the entire processing pipeline, including file combination, chunk creation, and vector database creation.",
            "expected_outcome": "All steps should be executed in the correct order without errors.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_initialize_mistral_model": {
            "description": "Tests the initialization of the Mistral model using an API key.",
            "expected_outcome": "The model should initialize correctly if the API key is present, and raise an error if the key is missing.",
            "status": "Passed" if result == 0 else "Failed"
        },
        "test_log_query": {
            "description": "Tests logging queries and answers to a JSON file.",
            "expected_outcome": "The query and answer should be logged correctly, and existing logs should be preserved.",
            "status": "Passed" if result == 0 else "Failed"
        }
    }
    
    # Write results to Excel
    write_test_results_to_excel(test_results)