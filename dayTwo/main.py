"""Main module for the deep learning-based recommender system.

This module provides the main entry point for running the recommender system,
including data loading, model training, evaluation, and visualization.
"""

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.data.processor import DataProcessor
from src.models.recommender import DeepRecommender
from src.training.trainer import ModelTrainer
from src.utils.logger import setup_logger

def main():
    """Run the deep learning-based recommender system pipeline."""
    # Set up logging
    logger = setup_logger('recommender')
    
    # Create necessary directories
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        data_processor = DataProcessor(data_dir)
        model = DeepRecommender(
            num_items=1000,  # Adjust based on your dataset
            embedding_dim=64,
            hidden_dim=128
        )
        trainer = ModelTrainer(model, learning_rate=0.001)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = data_processor.load_and_prepare_data()
        
        # Get session statistics
        stats = data_processor.get_session_statistics(df)
        logger.info(f"Session Statistics:")
        logger.info(f"Total sessions: {stats['total_sessions']}")
        logger.info(f"Total unique items: {stats['total_items']}")
        logger.info(f"Average session length: {stats['avg_session_length']:.2f}")
        
        # Create dataset and dataloader
        logger.info("Creating dataset and dataloader...")
        dataset = data_processor.create_dataset(df)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4
        )
        
        # Train the model
        logger.info("Training the model...")
        history = trainer.train(
            dataloader,
            num_epochs=10,
            validation_split=0.2
        )
        
        # Save the trained model
        logger.info("Saving the trained model...")
        model_path = model_dir / 'recommender_model.pt'
        trainer.save_model(model_path)
        
        # Evaluate the model
        logger.info("Evaluating the model...")
        test_loss = trainer.evaluate(dataloader)
        logger.info(f"Test Loss: {test_loss:.4f}")
        
        # Make some example recommendations
        logger.info("\nMaking example recommendations...")
        example_item = 1  # Replace with an actual item ID from your dataset
        recommendations = model.predict_next_items(
            torch.tensor([example_item]),
            top_k=5
        )
        
        logger.info(f"Top 5 recommendations for item {example_item}:")
        for item_id, prob in recommendations:
            logger.info(f"Item {item_id}: {prob:.4f}")
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == '__main__':
    main() 