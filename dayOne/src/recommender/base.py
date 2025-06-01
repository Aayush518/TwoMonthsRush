"""Base recommender class module.

This module contains the abstract base class for all recommender systems.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRecommender(ABC):
    """Abstract base class for recommender systems.
    
    This class defines the interface that all recommender systems must implement.
    It provides abstract methods for training and making recommendations.
    
    Attributes:
        is_fitted (bool): Whether the recommender has been trained.
    """
    
    def __init__(self):
        """Initialize the recommender."""
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, sessions: List[List[str]]) -> None:
        """Train the recommender on session data.
        
        Args:
            sessions: List of sessions, where each session is a list of item IDs.
        """
        pass
    
    @abstractmethod
    def recommend_next(self, current_item: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get recommendations for the next item given the current item.
        
        Args:
            current_item: The ID of the current item.
            top_k: Number of recommendations to return.
            
        Returns:
            List of tuples containing (item_id, probability) pairs.
            
        Raises:
            ValueError: If the recommender hasn't been trained.
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be trained before making recommendations")
    
    @abstractmethod
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the transition matrix used for recommendations.
        
        Returns:
            Dictionary mapping current items to dictionaries of next items and their probabilities.
            
        Raises:
            ValueError: If the recommender hasn't been trained.
        """
        if not self.is_fitted:
            raise ValueError("Recommender must be trained before getting transition matrix") 