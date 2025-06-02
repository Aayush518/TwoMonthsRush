"""Tests for the data processor module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.processor import DataProcessor

@pytest.fixture
def sample_data():
    """Create sample session data for testing."""
    data = {
        'session_id': [1, 1, 1, 2, 2, 3],
        'item_id': ['A', 'B', 'C', 'A', 'D', 'B']
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path

def test_data_processor_initialization(temp_data_dir):
    """Test DataProcessor initialization."""
    processor = DataProcessor(temp_data_dir)
    assert processor.data_dir == temp_data_dir

def test_load_and_prepare_data(temp_data_dir, sample_data):
    """Test data loading and preparation."""
    # Save sample data
    sample_data.to_csv(temp_data_dir / 'generated_sessions.csv', index=False)
    
    processor = DataProcessor(temp_data_dir)
    sessions, label_encoder = processor.load_and_prepare_data()
    
    # Check sessions
    assert len(sessions) == 3  # 3 unique sessions
    assert len(sessions[0]) == 3  # First session has 3 items
    assert len(sessions[1]) == 2  # Second session has 2 items
    assert len(sessions[2]) == 1  # Third session has 1 item
    
    # Check label encoder
    assert len(label_encoder.classes_) == 4  # 4 unique items
    assert all(item in label_encoder.classes_ for item in ['A', 'B', 'C', 'D'])

def test_get_session_statistics(sample_data):
    """Test session statistics calculation."""
    processor = DataProcessor(Path('dummy'))
    stats = processor.get_session_statistics(sample_data)
    
    assert stats['total_sessions'] == 3
    assert stats['total_items'] == 4
    assert np.isclose(stats['avg_session_length'], 2.0)  # (3 + 2 + 1) / 3

def test_load_and_prepare_data_empty_file(temp_data_dir):
    """Test handling of empty data file."""
    # Create empty file
    (temp_data_dir / 'generated_sessions.csv').touch()
    
    processor = DataProcessor(temp_data_dir)
    with pytest.raises(pd.errors.EmptyDataError):
        processor.load_and_prepare_data()

def test_load_and_prepare_data_missing_file(temp_data_dir):
    """Test handling of missing data file."""
    processor = DataProcessor(temp_data_dir)
    with pytest.raises(FileNotFoundError):
        processor.load_and_prepare_data() 