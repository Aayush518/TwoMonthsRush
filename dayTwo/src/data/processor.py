"""Data processing module for session data."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset

class SessionDataset(Dataset):
    """Dataset class for session data."""
    
    def __init__(self, sessions: List[List[int]], max_len: int = 10):
        self.sessions = sessions
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.sessions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        session = self.sessions[idx]
        
        # Pad or truncate session
        if len(session) > self.max_len:
            session = session[:self.max_len]
        elif len(session) < self.max_len:
            session = session + [0] * (self.max_len - len(session))
        
        # Convert to tensors
        x = torch.tensor(session[:-1], dtype=torch.long)
        y = torch.tensor(session[1:], dtype=torch.long)
        
        return x, y

class DataProcessor:
    """Class for processing session data."""
    
    def __init__(self, data_dir: Path):
        """Initialize the data processor.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.item_to_idx = {}
        self.idx_to_item = {}
        
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare the session data.
        
        Returns:
            DataFrame containing the processed session data
        """
        # Load the data
        df = pd.read_csv(self.data_dir / 'sample_sessions.csv')
        
        # Convert timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by session_id and timestamp
        df = df.sort_values(['session_id', 'timestamp'])
        
        # Create item mappings
        unique_items = df['item_id'].unique()
        self.item_to_idx = {item: idx + 1 for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx + 1: item for idx, item in enumerate(unique_items)}
        
        # Map item IDs to indices
        df['item_idx'] = df['item_id'].map(self.item_to_idx)
        
        return df
    
    def get_session_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the sessions.
        
        Args:
            df: DataFrame containing the session data
            
        Returns:
            Dictionary containing session statistics
        """
        stats = {
            'total_sessions': df['session_id'].nunique(),
            'total_items': len(self.item_to_idx),
            'avg_session_length': df.groupby('session_id').size().mean()
        }
        return stats
    
    def create_dataset(self, df: pd.DataFrame) -> SessionDataset:
        """Create a dataset from the session data.
        
        Args:
            df: DataFrame containing the session data
            
        Returns:
            SessionDataset instance
        """
        # Group by session_id and get item sequences
        sessions = df.groupby('session_id')['item_idx'].apply(list).tolist()
        
        return SessionDataset(sessions) 