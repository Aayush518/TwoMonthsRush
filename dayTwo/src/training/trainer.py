"""Model training module."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from pathlib import Path
import numpy as np

class ModelTrainer:
    """Class for training the recommender model."""
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = batch_y.view(-1)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Average loss on the evaluation data
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Reshape for loss calculation
                outputs = outputs.view(-1, outputs.size(-1))
                targets = batch_y.view(-1)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, dataloader: DataLoader, num_epochs: int = 10, validation_split: float = 0.2) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            dataloader: DataLoader for training data
            num_epochs: Number of epochs to train
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary containing training history
        """
        # Split data into train and validation
        dataset_size = len(dataloader.dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataloader.dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=dataloader.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Evaluate
            val_loss = self.evaluate(val_loader)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
        
        return history
    
    def save_model(self, path: Path) -> None:
        """Save the trained model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: Path) -> None:
        """Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 