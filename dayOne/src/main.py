"""Main module for the recommender system.

This module provides the main entry point for running the recommender system,
including data loading, training, evaluation, and visualization.
"""

import os
from pathlib import Path
from typing import List, Tuple

from data.processor import SessionDataProcessor
from recommender.markov_chain import MarkovChainRecommender
from visualization.visualizer import RecommenderVisualizer
from evaluation.evaluator import RecommenderEvaluator


def main():
    """Run the recommender system pipeline."""
    # Create necessary directories
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Initialize components
    data_processor = SessionDataProcessor(data_dir)
    recommender = MarkovChainRecommender()
    visualizer = RecommenderVisualizer(data_dir)
    evaluator = RecommenderEvaluator(recommender)
    
    try:
        # Load and prepare data
        print("Loading session data...")
        df = data_processor.load_sessions()
        
        print("\nPreparing sessions for training...")
        sessions = data_processor.prepare_sessions_for_training(df)
        
        # Train the recommender
        print("\nTraining Markov Chain recommender...")
        recommender.fit(sessions)
        
        # Create visualizations
        print("\nCreating visualizations...")
        visualizer.plot_transition_matrix(recommender.get_transition_matrix())
        visualizer.plot_transition_network(recommender.get_transition_matrix())
        visualizer.plot_session_statistics(df)
        
        # Evaluate the recommender
        print("\nEvaluating recommendations...")
        results = evaluator.evaluate_recommendations(sessions)
        evaluator.print_evaluation_results(results)
        
        # Print session statistics
        stats = data_processor.get_session_statistics(df)
        print("\nSession Statistics:")
        print(f"Total sessions: {stats['total_sessions']}")
        print(f"Total unique items: {stats['total_items']}")
        print(f"Average session length: {stats['avg_session_length']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
