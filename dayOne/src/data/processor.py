"""Data processor module.

This module handles loading and processing of session data for the recommender system.
"""

from typing import List, Dict, Any
import pandas as pd
from pathlib import Path


class SessionDataProcessor:
    """Class for processing session data.
    
    This class handles loading session data from CSV files and preparing it
    for training the recommender system.
    
    Attributes:
        data_path (Path): Path to the data directory.
    """
    
    def __init__(self, data_path: str = 'data'):
        """Initialize the data processor.
        
        Args:
            data_path: Path to the data directory.
        """
        self.data_path = Path(data_path)
    
    def load_sessions(self, csv_file: str = 'generated_sessions.csv') -> pd.DataFrame:
        """Load session data from a CSV file.
        
        Args:
            csv_file: Name of the CSV file in the data directory.
            
        Returns:
            DataFrame containing the session data.
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist.
        """
        file_path = self.data_path / csv_file
        if not file_path.exists():
            raise FileNotFoundError(f"Session data file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def prepare_sessions_for_training(self, df: pd.DataFrame) -> List[List[str]]:
        """Convert DataFrame to list of item sequences for training.
        
        Args:
            df: DataFrame containing session data.
            
        Returns:
            List of sessions, where each session is a list of item IDs.
        """
        df = df.sort_values(['session_id', 'timestamp'])
        sessions = []
        for session_id, group in df.groupby('session_id'):
            session_items = group['item_id'].tolist()
            sessions.append(session_items)
        return sessions
    
    def get_session_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the sessions.
        
        Args:
            df: DataFrame containing session data.
            
        Returns:
            Dictionary containing various session statistics.
        """
        stats = {
            'total_sessions': df['session_id'].nunique(),
            'total_items': df['item_id'].nunique(),
            'avg_session_length': df.groupby('session_id').size().mean(),
            'interaction_types': df['interaction_type'].value_counts().to_dict(),
            'device_types': df['device_type'].value_counts().to_dict()
        }
        return stats 