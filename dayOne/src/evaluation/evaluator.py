"""Evaluation module.

This module handles the evaluation of recommender system performance.
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict
from recommender.base import BaseRecommender


class RecommenderEvaluator:
    """Class for evaluating recommender system performance.
    
    This class provides methods for evaluating the performance of a recommender
    system using various metrics such as hit rate and item popularity.
    
    Attributes:
        recommender (BaseRecommender): The recommender system to evaluate.
    """
    
    def __init__(self, recommender: BaseRecommender):
        """Initialize the evaluator.
        
        Args:
            recommender: The recommender system to evaluate.
        """
        self.recommender = recommender
    
    def evaluate_recommendations(
        self,
        test_sessions: List[List[str]],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Evaluate recommendation performance.
        
        Args:
            test_sessions: List of test sessions.
            top_k: Number of recommendations to consider.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        hits = defaultdict(int)
        total_predictions = defaultdict(int)
        item_popularity = defaultdict(int)
        
        for session in test_sessions:
            if len(session) < 2:
                continue
                
            current_item = session[-2]
            actual_next = session[-1]
            
            # Update item popularity
            item_popularity[actual_next] += 1
            
            # Get recommendations
            recommendations = self.recommender.recommend_next(current_item, top_k=top_k)
            recommended_items = [item for item, _ in recommendations]
            
            # Update hit counts
            if actual_next in recommended_items:
                hits[current_item] += 1
            total_predictions[current_item] += 1
        
        # Calculate hit rates
        hit_rates = {}
        for item in total_predictions:
            hit_rates[item] = hits[item] / total_predictions[item]
        
        # Calculate overall metrics
        total_hits = sum(hits.values())
        total_predictions_made = sum(total_predictions.values())
        overall_hit_rate = total_hits / total_predictions_made if total_predictions_made > 0 else 0
        
        # Get top items by hit rate and popularity
        top_predictable = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        top_popular = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_sessions': len(test_sessions),
            'total_predictions': total_predictions_made,
            'overall_hit_rate': overall_hit_rate,
            'top_predictable_items': [
                {'item': item, 'hit_rate': rate, 'hits': hits[item], 'total': total_predictions[item]}
                for item, rate in top_predictable
            ],
            'top_popular_items': [
                {'item': item, 'occurrences': count}
                for item, count in top_popular
            ]
        }
    
    def print_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Print evaluation results in a readable format.
        
        Args:
            results: Dictionary containing evaluation results.
        """
        print("\nDetailed Recommendation Analysis:")
        print(f"Total test sessions: {results['total_sessions']}")
        print(f"Total predictions made: {results['total_predictions']}")
        print(f"Overall hit rate: {results['overall_hit_rate']:.2%}")
        
        print("\nTop 5 most predictable items (highest hit rate):")
        for item_data in results['top_predictable_items']:
            print(f"{item_data['item']}: {item_data['hit_rate']:.2%} "
                  f"({item_data['hits']}/{item_data['total']})")
        
        print("\nTop 5 most popular items:")
        for item_data in results['top_popular_items']:
            print(f"{item_data['item']}: {item_data['occurrences']} occurrences") 