# Deep Learning Based Recommender System

This project implements a deep learning based recommender system using PyTorch. The system uses neural embeddings and feed-forward networks to learn item relationships from user sessions.

## Features

- Neural embeddings to represent items in a dense vector space
- Multi-layer feed-forward network for learning item relationships
- Session-based data processing with padding and truncation
- Comprehensive training pipeline with validation
- Model evaluation and recommendation generation

## Project Structure

```
dayTwo/
├── data/
│   └── sample_sessions.csv    # Sample session data
├── src/
│   ├── models/
│   │   └── recommender.py     # DeepRecommender model implementation
│   ├── data/
│   │   └── processor.py       # Data processing and dataset creation
│   ├── training/
│   │   └── trainer.py         # Model training and evaluation
│   └── utils/
│       └── logger.py          # Logging utilities
├── tests/                     # Comprehensive test suite
├── main.py                    # Main entry point
└── requirements.txt           # Project dependencies
```

## Model Architecture

The recommender system uses a deep neural network with the following components:

- **Embedding Layer**: Maps item IDs to dense vectors (64 dimensions)
- **Neural Network**:
  - Input: Item embeddings
  - Hidden layers: Two fully connected layers (128 dimensions)
  - Output: Probability distribution over items
  - Dropout for regularization

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the recommender system:
```bash
python main.py
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

## Dependencies

- PyTorch >= 2.0.0
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- pytest-catchlog >= 1.2.2 