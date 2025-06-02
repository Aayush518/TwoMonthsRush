"""Dataset module for the recommender system."""

import torch
from torch.utils.data import Dataset
from typing import List

class SessionDataset(Dataset):
    """Custom Dataset for session data."""
    def __init__(self, sessions: List[List[int]], max_len: int = 20):
        """
        Initialize the dataset.
        
        Args:
            sessions: List of session sequences
            max_len: Maximum sequence length
        """
        self.sessions = sessions
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        session = self.sessions[idx]
        # Pad or truncate session to max_len
        if len(session) < self.max_len:
            session = session + [0] * (self.max_len - len(session))
        else:
            session = session[:self.max_len]
        return torch.tensor(session[:-1]), torch.tensor(session[1:]) 