"""Tests for the dataset module."""

import pytest
import torch
from src.data.dataset import SessionDataset

def test_session_dataset_initialization():
    """Test dataset initialization."""
    sessions = [[1, 2, 3], [4, 5, 6, 7]]
    dataset = SessionDataset(sessions)
    assert len(dataset) == 2
    assert dataset.max_len == 20  # default value

def test_session_dataset_getitem():
    """Test dataset item retrieval."""
    sessions = [[1, 2, 3], [4, 5, 6, 7]]
    dataset = SessionDataset(sessions, max_len=5)
    
    # Test first session
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (4,)  # max_len - 1
    assert y.shape == (4,)  # max_len - 1
    assert torch.equal(x, torch.tensor([1, 2, 3, 0]))
    assert torch.equal(y, torch.tensor([2, 3, 0, 0]))
    
    # Test second session
    x, y = dataset[1]
    assert torch.equal(x, torch.tensor([4, 5, 6, 7]))
    assert torch.equal(y, torch.tensor([5, 6, 7, 0]))

def test_session_dataset_padding():
    """Test session padding behavior."""
    sessions = [[1, 2, 3]]
    dataset = SessionDataset(sessions, max_len=5)
    x, y = dataset[0]
    assert len(x) == 4  # max_len - 1
    assert len(y) == 4  # max_len - 1
    assert torch.equal(x, torch.tensor([1, 2, 3, 0]))
    assert torch.equal(y, torch.tensor([2, 3, 0, 0]))

def test_session_dataset_truncation():
    """Test session truncation behavior."""
    sessions = [[1, 2, 3, 4, 5, 6]]
    dataset = SessionDataset(sessions, max_len=4)
    x, y = dataset[0]
    assert len(x) == 3  # max_len - 1
    assert len(y) == 3  # max_len - 1
    assert torch.equal(x, torch.tensor([1, 2, 3]))
    assert torch.equal(y, torch.tensor([2, 3, 4])) 