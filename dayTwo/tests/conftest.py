"""Shared test fixtures."""

import pytest
import torch
import pandas as pd
from pathlib import Path
from src.model.recommender import DeepRecommender
from src.training.trainer import ModelTrainer

@pytest.fixture
def sample_session_data():
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

@pytest.fixture
def test_model():
    """Create a test model instance."""
    return DeepRecommender(num_items=10, embedding_dim=4, hidden_dim=8)

@pytest.fixture
def test_trainer(test_model):
    """Create a test trainer instance."""
    return ModelTrainer(test_model, learning_rate=0.001)

@pytest.fixture
def sample_tensor_data():
    """Create sample tensor data for testing."""
    x = torch.randint(0, 10, (10, 5))  # 10 samples, sequence length 5
    y = torch.randint(0, 10, (10, 5))
    return x, y 