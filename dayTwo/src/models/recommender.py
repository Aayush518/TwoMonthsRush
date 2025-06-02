"""Deep learning-based recommender model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class DeepRecommender(nn.Module):
    """Deep learning-based recommender model.
    
    This model uses embeddings and neural networks to learn item relationships
    and make recommendations.
    
    Args:
        num_items: Number of unique items in the dataset
        embedding_dim: Dimension of item embeddings
        hidden_dim: Dimension of hidden layers
    """
    
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        
        # Item embeddings
        self.item_embeddings = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # Neural network layers
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_items + 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size,)
            
        Returns:
            Output tensor of shape (batch_size, num_items + 1)
        """
        # Get item embeddings
        x = self.item_embeddings(x)
        
        # Pass through neural network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def predict_next_items(self, current_items: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
        """Predict the next items given current items.
        
        Args:
            current_items: Tensor of current item IDs
            top_k: Number of top recommendations to return
            
        Returns:
            List of (item_id, probability) tuples
        """
        self.eval()
        with torch.no_grad():
            # Get model predictions
            logits = self(current_items)
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k items
            top_probs, top_indices = torch.topk(probs, top_k)
            
            # Convert to list of tuples
            recommendations = []
            for idx, prob in zip(top_indices[0], top_probs[0]):
                recommendations.append((idx.item(), prob.item()))
            
            return recommendations 