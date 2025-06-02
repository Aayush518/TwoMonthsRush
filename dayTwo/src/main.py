"""Main entry point for the recommender system."""

import os
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from data.processor import DataProcessor
from data.dataset import SessionDataset
from model.recommender import DeepRecommender
from training.trainer import ModelTrainer
from utils.logger import setup_logger

def main():
    """Run the recommender system pipeline."""
    # Setup logging
    logger = setup_logger('recommender')
    
    try:
        # Create necessary directories
        data_dir = Path('data')
        data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        data_processor = DataProcessor(data_dir)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        sessions, label_encoder = data_processor.load_and_prepare_data()
        
        # Split data
        train_sessions, val_sessions = train_test_split(
            sessions, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = SessionDataset(train_sessions)
        val_dataset = SessionDataset(val_sessions)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model
        num_items = len(label_encoder.classes_)
        model = DeepRecommender(num_items)
        
        # Initialize trainer
        trainer = ModelTrainer(model)
        
        # Train model
        logger.info("Starting model training...")
        history = trainer.train(train_loader, val_loader)
        
        # Save model
        model_path = data_dir / 'deep_recommender.pth'
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 