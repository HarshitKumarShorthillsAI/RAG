import os
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import patch, MagicMock
import tempfile
from datetime import datetime

# Import the functions from your script
# Use the correct module name that matches the import in the original script
from evaluation_main import (
    calculate_metrics, 
    log_interaction, 
    load_test_cases, 
    qa_pipeline, 
    calculate_grade,
    setup_logger,
    LOG_FILE,  # Import LOG_FILE from the module
    MISTRAL_MODEL,  # Import MISTRAL_MODEL for testing
    VALID_MODELS    # Import VALID_MODELS for testing
)

# Test fixture for temporary files
@pytest.fixture
def temp_files():
    with tempfile.NamedTemporaryFile(suffix='.xlsx') as test_cases_file11, \
         tempfile.NamedTemporaryFile(suffix='.log') as log_file22, \
         tempfile.NamedTemporaryFile(suffix='.xlsx') as results_file22:
        
        # Create a simple test dataframe
        df = pd.DataFrame({
            'Question': ['What is Python?', 'Who created Python?'],
            'Answer': ['Python is a programming language.', 'Guido van Rossum created Python.'],
            'Context': ['Python is a high-level language.', 'Python was created in the 1990s.']
        })
        
        # Save to temp file
        df.to_excel(test_cases_file11.name, index=False)
        
        yield {
            'test_cases_file': test_cases_file11.name,
            'log_file': log_file22.name,
            'results_file': results_file22.name
        }

# Test Case 1: Test metric calculation function
def test_calculate_metrics():
    # Test with identical texts
    identical_metrics = calculate_metrics(
        "Python is a programming language.", 
        "Python is a programming language."
    )
    assert identical_metrics['rouge_score'] == 1.0
    assert identical_metrics['cosine_similarity'] > 0.99
    assert identical_metrics['bert_score_f1'] > 0.99
    
    # Test with completely different texts
    different_metrics = calculate_metrics(
        "Python is a programming language.", 
        "JavaScript is used for web development."
    )
    assert different_metrics['rouge_score'] < 0.5
    assert different_metrics['cosine_similarity'] < 0.8
    
    # Test with partially similar texts
    partial_metrics = calculate_metrics(
        "Python is a high-level programming language.", 
        "Python is a programming language used for various purposes."
    )
    assert 0.3 < partial_metrics['rouge_score'] < 1.0
    assert 0.7 < partial_metrics['cosine_similarity'] < 1.0

# Test Case 2: Test grade calculation function
def test_calculate_grade():
    assert calculate_grade(0.95) == "A (Excellent)"
    assert calculate_grade(0.85) == "B (Good)"
    assert calculate_grade(0.75) == "C (Average)"
    assert calculate_grade(0.65) == "D (Below Average)"
    assert calculate_grade(0.55) == "F (Poor)"

# Test Case 3: Test logging functionality
def test_log_interaction(temp_files):
    # Use the temporary log file instead of the global LOG_FILE
    with patch('evaluation_main.LOG_FILE', temp_files['log_file']):
        # Ensure the logger is set up with the temp log file
        setup_logger()
        
        metrics = {
            'rouge_score': 0.85,
            'cosine_similarity': 0.90,
            'bert_score_precision': 0.92,
            'bert_score_recall': 0.88,
            'bert_score_f1': 0.90,
            'final_score': 0.88
        }
        
        log_interaction(
            question="What is Python?",
            context="Python is a programming language.",
            generated="Python is a high-level programming language.",
            reference="Python is a versatile programming language.",
            metrics=metrics
        )
        
        # Verify log file was written correctly
        with open(temp_files['log_file'], 'r') as f:
            log_content = f.read()
            assert "What is Python?" in log_content
            assert "rouge_score" in log_content
            assert "final_score" in log_content

# Test Case 4: Test loading test cases
def test_load_test_cases(temp_files):
    df = load_test_cases(temp_files['test_cases_file'])
    
    assert not df.empty
    assert 'Question' in df.columns
    assert 'Answer' in df.columns
    assert 'Context' in df.columns
    assert len(df) == 2

# Test Case 5: Test API interaction with mocked response
@patch('requests.post')
def test_qa_pipeline(mock_post):
    # Mock response object
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language."
                }
            }
        ]
    }
    mock_post.return_value = mock_response
    
    answer = qa_pipeline("What is Python?", "Python is a high-level language.")
    assert answer == "Python is a programming language."
    
    # Verify the API was called with correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args[1]['json']
    assert "What is Python?" in call_args['messages'][1]['content']
    assert "Python is a high-level language" in call_args['messages'][1]['content']

# Test Case 6: Test API error handling
@patch('requests.post')
def test_qa_pipeline_error_handling(mock_post):
    # First response: rate limit error
    rate_limit_response = MagicMock()
    rate_limit_response.status_code = 429
    
    # Second response: success
    success_response = MagicMock()
    success_response.status_code = 200
    success_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Python is a programming language."
                }
            }
        ]
    }
    
    # Configure mock to return different responses on consecutive calls
    mock_post.side_effect = [rate_limit_response, success_response]
    
    # Patch sleep to avoid waiting during tests
    with patch('time.sleep'):
        answer = qa_pipeline("What is Python?", "Python is a high-level language.")
    
    assert answer == "Python is a programming language."
    assert mock_post.call_count == 2  # Called twice due to retry
