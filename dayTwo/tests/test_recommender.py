"""Tests for the recommender model module."""

import pytest
import torch
from src.model.recommender import DeepRecommender

@pytest.fixture
def model():
    """Create a test model instance."""
    return DeepRecommender(num_items=10, embedding_dim=4, hidden_dim=8)

def test_model_initialization():
    """Test model initialization."""
    model = DeepRecommender(num_items=10)
    assert model.embedding.num_embeddings == 11  # num_items + 1 for padding
    assert model.embedding.embedding_dim == 64  # default value
    assert model.lstm.hidden_size == 128  # default value
    assert model.fc.out_features == 11  # num_items + 1

def test_model_forward(model):
    """Test model forward pass."""
    batch_size = 2
    seq_length = 3
    x = torch.randint(0, 10, (batch_size, seq_length))
    
    output = model(x)
    assert output.shape == (batch_size, seq_length, 11)  # num_items + 1

def test_model_predict_next_items(model):
    """Test next item prediction."""
    batch_size = 2
    seq_length = 3
    x = torch.randint(0, 10, (batch_size, seq_length))
    
    top_indices, top_probs = model.predict_next_items(x, top_k=3)
    assert top_indices.shape == (batch_size, 3)
    assert top_probs.shape == (batch_size, 3)
    assert torch.all(top_probs >= 0) and torch.all(top_probs <= 1)
    assert torch.all(top_indices >= 0) and torch.all(top_indices < 11)

def test_model_padding_handling(model):
    """Test model handling of padding tokens."""
    x = torch.tensor([[0, 1, 2], [1, 0, 2]])  # 0 is padding token
    output = model(x)
    assert output.shape == (2, 3, 11)

def test_model_gradient_flow(model):
    """Test that gradients flow properly during training."""
    x = torch.randint(0, 10, (2, 3))
    y = torch.randint(0, 10, (2, 3))
    
    output = model(x)
    loss = torch.nn.functional.cross_entropy(
        output.view(-1, 11),
        y.view(-1)
    )
    loss.backward()
    
    # Check if gradients exist and are not None
    for param in model.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()
        assert not torch.isinf(param.grad).any() 