# Day Two Challenge - Deep Learning Recommender System

## Description
This project implements a state-of-the-art deep learning based recommender system using PyTorch. Unlike traditional recommender systems, this implementation uses LSTM (Long Short-Term Memory) networks to capture complex sequential patterns in user sessions.

## Features
- Deep Learning based recommendation system using LSTM
- Session sequence modeling
- Embedding layer for item representation
- Training and validation pipeline
- Model checkpointing
- Comprehensive logging

## Project Structure
```
dayTwo/
├── src/                   # Source code
│   ├── data/             # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py    # Custom dataset implementation
│   │   └── processor.py  # Data loading and preprocessing
│   ├── model/            # Model implementation
│   │   ├── __init__.py
│   │   └── recommender.py # Deep learning model
│   ├── training/         # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py    # Model training and evaluation
│   ├── utils/            # Utility functions
│   │   ├── __init__.py
│   │   └── logger.py     # Logging utilities
│   ├── __init__.py
│   └── main.py          # Main entry point
├── data/                 # Data directory
│   ├── generated_sessions.csv
│   └── deep_recommender.pth
├── tests/               # Test files
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Technical Details
- Uses PyTorch for deep learning implementation
- LSTM architecture for sequence modeling
- Embedding layer for item representation
- Cross-entropy loss for training
- Adam optimizer
- GPU support (if available)

## Setup
1. Make sure you have Python 3.x installed
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Solution
To run the solution, execute:
```bash
python src/main.py
```

This will:
1. Load and preprocess the session data
2. Create train and validation datasets
3. Initialize and train the deep learning model
4. Save the trained model

## Model Architecture
The model consists of:
1. Embedding Layer: Converts item IDs to dense vectors
2. LSTM Layer: Processes sequential session data
3. Fully Connected Layer: Predicts next items

## Performance
The model is evaluated using:
- Training loss
- Validation loss
- Per-epoch metrics

## Requirements
- Python 3.x
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- tqdm 