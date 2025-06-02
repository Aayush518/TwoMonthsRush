"""Tests for the training module."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model.recommender import DeepRecommender
from src.training.trainer import ModelTrainer

@pytest.fixture
def model():
    """Create a test model instance."""
    return DeepRecommender(num_items=10, embedding_dim=4, hidden_dim=8)

@pytest.fixture
def trainer(model):
    """Create a test trainer instance."""
    return ModelTrainer(model, learning_rate=0.001)

@pytest.fixture
def sample_data():
    """Create sample training data."""
    x = torch.randint(0, 10, (10, 5))  # 10 samples, sequence length 5
    y = torch.randint(0, 10, (10, 5))
    return TensorDataset(x, y)

@pytest.fixture
def train_loader(sample_data):
    """Create a test data loader."""
    return DataLoader(sample_data, batch_size=2)

def test_trainer_initialization(model):
    """Test trainer initialization."""
    trainer = ModelTrainer(model)
    assert trainer.device in ['cuda', 'cpu']
    assert isinstance(trainer.criterion, torch.nn.CrossEntropyLoss)
    assert isinstance(trainer.optimizer, torch.optim.Adam)

def test_train_epoch(trainer, train_loader):
    """Test training for one epoch."""
    initial_loss = trainer.evaluate(train_loader)
    epoch_loss = trainer.train_epoch(train_loader)
    
    assert isinstance(epoch_loss, float)
    assert epoch_loss >= 0
    # Note: We don't assert that loss decreased as it might not in a single epoch

def test_evaluate(trainer, train_loader):
    """Test model evaluation."""
    val_loss = trainer.evaluate(train_loader)
    assert isinstance(val_loss, float)
    assert val_loss >= 0

def test_train(trainer, train_loader):
    """Test full training process."""
    history = trainer.train(train_loader, train_loader, num_epochs=2)
    
    assert isinstance(history, dict)
    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2

def test_save_and_load_model(trainer, tmp_path):
    """Test model saving and loading."""
    # Save model
    save_path = tmp_path / "test_model.pth"
    trainer.save_model(str(save_path))
    assert save_path.exists()
    
    # Load model
    trainer.load_model(str(save_path))
    # If we get here without errors, loading was successful

def test_trainer_device_handling(model):
    """Test trainer device handling."""
    # Test CPU device
    trainer_cpu = ModelTrainer(model, device='cpu')
    assert trainer_cpu.device == 'cpu'
    
    # Test CUDA device if available
    if torch.cuda.is_available():
        trainer_cuda = ModelTrainer(model, device='cuda')
        assert trainer_cuda.device == 'cuda' 