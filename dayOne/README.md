# Markov Chain Recommender System

A recommender system that uses Markov Chains to model item transitions in user sessions and make recommendations.

## Project Structure

```
day1/
├── data/                  # Data directory
│   ├── generated_sessions.csv
│   ├── transition_matrix.png
│   ├── transition_network.png
│   └── session_statistics/
├── src/                   # Source code
│   ├── data/             # Data processing
│   │   ├── __init__.py
│   │   └── processor.py
│   ├── recommender/      # Recommender implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── markov_chain.py
│   ├── visualization/    # Visualization tools
│   │   ├── __init__.py
│   │   └── visualizer.py
│   ├── evaluation/       # Evaluation tools
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── __init__.py
│   └── main.py
├── tests/                # Test files
│   └── test_recommender.py
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Features

- Markov Chain-based recommendation system
- Session data processing and analysis
- Transition matrix and network visualization
- Performance evaluation metrics
- Comprehensive documentation

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate session data (if not already present):
```bash
python src/main.py
```

2. Run the recommender system:
```bash
python src/main.py
```

This will:
- Load and process session data
- Train the Markov Chain recommender
- Create visualizations
- Evaluate the system's performance

## Components

### Data Processing
- `SessionDataProcessor`: Handles loading and processing of session data

### Recommender
- `BaseRecommender`: Abstract base class for recommender systems
- `MarkovChainRecommender`: Implementation using Markov Chains

### Visualization
- `RecommenderVisualizer`: Creates various visualizations:
  - Transition matrix heatmap
  - Transition network graph
  - Session statistics plots

### Evaluation
- `RecommenderEvaluator`: Evaluates recommender performance using:
  - Hit rate
  - Item popularity
  - Predictability metrics

## Development

To run tests:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# 2MonthsRush
