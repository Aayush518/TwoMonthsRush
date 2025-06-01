"""Visualization module.

This module handles the creation of various visualizations for the recommender system.
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from pathlib import Path
import pandas as pd


class RecommenderVisualizer:
    """Class for creating visualizations of the recommender system.
    
    This class handles the creation of various visualizations including
    transition matrices, network graphs, and session statistics.
    
    Attributes:
        output_dir (Path): Directory to save visualizations.
    """
    
    def __init__(self, output_dir: str = 'data'):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_transition_matrix(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        output_file: str = 'transition_matrix.png'
    ) -> None:
        """Create a heatmap of the transition matrix.
        
        Args:
            transition_matrix: Matrix of transition probabilities.
            output_file: Name of the output file.
        """
        # Get all unique items
        items = sorted(list(transition_matrix.keys()))
        
        # Create the matrix
        matrix = np.zeros((len(items), len(items)))
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items):
                matrix[i, j] = transition_matrix[item1].get(item2, 0)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, 
                    xticklabels=items,
                    yticklabels=items,
                    cmap='YlOrRd',
                    annot=True,
                    fmt='.2f',
                    square=True)
        plt.title('Item Transition Probabilities')
        plt.xlabel('Next Item')
        plt.ylabel('Current Item')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file)
        plt.close()
        print(f"Transition matrix heatmap saved to {self.output_dir / output_file}")
    
    def plot_transition_network(
        self,
        transition_matrix: Dict[str, Dict[str, float]],
        output_file: str = 'transition_network.png'
    ) -> None:
        """Create a network graph of transitions.
        
        Args:
            transition_matrix: Matrix of transition probabilities.
            output_file: Name of the output file.
        """
        G = nx.DiGraph()
        
        # Add edges with weights
        for item1, transitions in transition_matrix.items():
            for item2, prob in transitions.items():
                if prob > 0.1:  # Only show significant transitions
                    G.add_edge(item1, item2, weight=prob)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 20))
        
        # Use a circular layout
        pos = nx.circular_layout(G)
        
        # Calculate node sizes based on in-degree
        in_degrees = dict(G.in_degree())
        node_sizes = [1000 + in_degrees[node] * 200 for node in G.nodes()]
        
        # Color nodes based on their role
        out_degrees = dict(G.out_degree())
        node_colors = []
        for node in G.nodes():
            if in_degrees[node] > out_degrees[node]:
                node_colors.append('lightgreen')  # Popular destinations
            elif out_degrees[node] > in_degrees[node]:
                node_colors.append('lightblue')   # Good recommenders
            else:
                node_colors.append('lightgray')   # Balanced nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_size=node_sizes,
                             node_color=node_colors,
                             alpha=0.8,
                             ax=ax)
        
        # Draw edges
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
        edge_cmap = plt.cm.Blues
        
        edges = nx.draw_networkx_edges(G, pos,
                                     width=edge_weights,
                                     edge_color=edge_colors,
                                     edge_cmap=edge_cmap,
                                     alpha=0.6,
                                     arrows=True,
                                     arrowsize=20,
                                     ax=ax)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Transition Probability')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos,
                              font_size=12,
                              font_weight='bold',
                              bbox=dict(facecolor='white',
                                      edgecolor='none',
                                      alpha=0.7),
                              ax=ax)
        
        # Add title and legend
        ax.set_title('Item Transition Network\n' +
                    'Green: Popular Destinations | Blue: Good Recommenders | Gray: Balanced',
                    fontsize=16, pad=20)
        
        # Add description
        plt.figtext(0.5, 0.01,
                   'Node size indicates popularity as a destination\n' +
                   'Edge thickness and color indicate transition probability\n' +
                   'Only transitions with probability > 0.1 are shown',
                   ha='center', fontsize=12)
        
        # Add node role statistics
        role_stats = {
            'Popular Destinations': sum(1 for c in node_colors if c == 'lightgreen'),
            'Good Recommenders': sum(1 for c in node_colors if c == 'lightblue'),
            'Balanced': sum(1 for c in node_colors if c == 'lightgray')
        }
        
        stats_text = '\n'.join([f"{role}: {count} items" for role, count in role_stats.items()])
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Transition network graph saved to {self.output_dir / output_file}")
    
    def plot_session_statistics(self, df: pd.DataFrame) -> None:
        """Plot various session statistics.
        
        Args:
            df: DataFrame containing session data.
        """
        # Session length distribution
        session_lengths = df.groupby('session_id').size()
        plt.figure(figsize=(10, 6))
        sns.histplot(session_lengths, bins=20)
        plt.title('Session Length Distribution')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Count')
        plt.savefig(self.output_dir / 'session_lengths.png')
        plt.close()
        
        # Interaction type distribution
        plt.figure(figsize=(10, 6))
        df['interaction_type'].value_counts().plot(kind='bar')
        plt.title('Interaction Type Distribution')
        plt.xlabel('Interaction Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'interaction_types.png')
        plt.close()
        
        # Device type distribution
        plt.figure(figsize=(10, 6))
        df['device_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('Device Type Distribution')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'device_types.png')
        plt.close()
        
        print(f"Session statistics plots saved to {self.output_dir}/") 