"""Markov Chain recommender module.

This module implements a Markov Chain-based recommender system that uses
transition probabilities between items to make recommendations.
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

from .base import BaseRecommender


class MarkovChainRecommender(BaseRecommender):
    """Markov Chain-based recommender system.
    
    This recommender uses a first-order Markov Chain to model item transitions
    in user sessions. It makes recommendations based on the probability of
    transitioning from the current item to other items.
    
    Attributes:
        transition_matrix (Dict[str, Dict[str, float]]): Matrix of transition probabilities.
        is_fitted (bool): Whether the recommender has been trained.
    """
    
    def __init__(self):
        """Initialize the Markov Chain recommender."""
        super().__init__()
        self.transition_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    def fit(self, sessions: List[List[str]]) -> None:
        """Train the recommender on session data.
        
        This method builds the transition matrix by counting transitions between
        items in the training sessions and normalizing the counts to get probabilities.
        
        Args:
            sessions: List of sessions, where each session is a list of item IDs.
        """
        # Count transitions
        for session in sessions:
            for i in range(len(session) - 1):
                current_item = session[i]
                next_item = session[i + 1]
                self.transition_matrix[current_item][next_item] += 1
        
        # Normalize to get probabilities
        for current_item in self.transition_matrix:
            total = sum(self.transition_matrix[current_item].values())
            if total > 0:
                for next_item in self.transition_matrix[current_item]:
                    self.transition_matrix[current_item][next_item] /= total
        
        self.is_fitted = True
    
    def recommend_next(self, current_item: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get recommendations for the next item given the current item.
        
        Args:
            current_item: The ID of the current item.
            top_k: Number of recommendations to return.
            
        Returns:
            List of tuples containing (item_id, probability) pairs, sorted by probability.
            
        Raises:
            ValueError: If the recommender hasn't been trained or if current_item
                       hasn't been seen in training.
        """
        super().recommend_next(current_item, top_k)
        
        if current_item not in self.transition_matrix:
            raise ValueError(f"Item {current_item} not seen in training data")
        
        # Get transition probabilities for current item
        transitions = self.transition_matrix[current_item]
        
        # Sort by probability and return top_k
        sorted_transitions = sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_transitions[:top_k]
    
    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the transition matrix used for recommendations.
        
        Returns:
            Dictionary mapping current items to dictionaries of next items and their probabilities.
            
        Raises:
            ValueError: If the recommender hasn't been trained.
        """
        super().get_transition_matrix()
        return dict(self.transition_matrix) 