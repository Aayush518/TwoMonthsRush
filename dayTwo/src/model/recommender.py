"""Deep learning recommender model module."""

import torch
import torch.nn as nn
from typing import Tuple

class DeepRecommender(nn.Module):
    """Deep Learning based Recommender System."""
    
    def __init__(self, num_items: int, embedding_dim: int = 64, hidden_dim: int = 128):
        """
        Initialize the recommender model.
        
        Args:
            num_items: Number of unique items
            embedding_dim: Dimension of item embeddings
            hidden_dim: Dimension of LSTM hidden state
        """
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, num_items + 1)
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out)
    
    def predict_next_items(self, session: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next items for a given session.
        
        Args:
            session: Input session tensor
            top_k: Number of top items to return
            
        Returns:
            Tuple containing:
            - Top k item indices
            - Top k item probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self(session)
            # Get predictions for the last item in sequence
            last_output = outputs[:, -1, :]
            probs = torch.softmax(last_output, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k)
        return top_indices, top_probs 