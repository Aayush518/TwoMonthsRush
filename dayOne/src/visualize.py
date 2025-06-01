import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from recommender.markov_chain_recommender import MarkovChainRecommender
import networkx as nx
from collections import defaultdict

def load_sessions(csv_file='data/generated_sessions.csv'):
    """Load and prepare session data."""
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def prepare_sessions_for_training(df):
    """Convert DataFrame to list of item sequences for training."""
    df = df.sort_values(['session_id', 'timestamp'])
    sessions = []
    for session_id, group in df.groupby('session_id'):
        session_items = group['item_id'].tolist()
        sessions.append(session_items)
    return sessions

def plot_transition_matrix(transition_matrix, output_file='data/transition_matrix.png'):
    """Create a heatmap of the transition matrix."""
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
    plt.savefig(output_file)
    plt.close()
    print(f"Transition matrix heatmap saved to {output_file}")

def plot_transition_network(transition_matrix, output_file='data/transition_network.png'):
    """Create a network graph of transitions with improved visualization."""
    G = nx.DiGraph()
    
    # Add edges with weights
    for item1, transitions in transition_matrix.items():
        for item2, prob in transitions.items():
            if prob > 0.1:  # Only show significant transitions
                G.add_edge(item1, item2, weight=prob)
    
    # Create the plot with a larger figure size
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Use a circular layout for better organization
    pos = nx.circular_layout(G)
    
    # Calculate node sizes based on in-degree (popularity as destination)
    in_degrees = dict(G.in_degree())
    node_sizes = [1000 + in_degrees[node] * 200 for node in G.nodes()]
    
    # Draw nodes with different colors based on their role
    # Nodes with high in-degree are destinations (popular items)
    # Nodes with high out-degree are sources (good recommenders)
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
    
    # Draw edges with varying widths and colors based on probability
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Create a colormap for edges
    edge_cmap = plt.cm.Blues
    
    # Draw edges with a colormap
    edges = nx.draw_networkx_edges(G, pos,
                                 width=edge_weights,
                                 edge_color=edge_colors,
                                 edge_cmap=edge_cmap,
                                 alpha=0.6,
                                 arrows=True,
                                 arrowsize=20,
                                 ax=ax)
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Transition Probability')
    
    # Draw labels with a white background for better readability
    nx.draw_networkx_labels(G, pos,
                           font_size=12,
                           font_weight='bold',
                           bbox=dict(facecolor='white',
                                   edgecolor='none',
                                   alpha=0.7),
                           ax=ax)
    
    # Add a title and legend
    ax.set_title('Item Transition Network\n' +
                'Green: Popular Destinations | Blue: Good Recommenders | Gray: Balanced',
                fontsize=16, pad=20)
    
    # Add a description
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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Transition network graph saved to {output_file}")

def plot_session_statistics(df, output_dir='data'):
    """Plot various session statistics."""
    # Session length distribution
    session_lengths = df.groupby('session_id').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(session_lengths, bins=20)
    plt.title('Session Length Distribution')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/session_lengths.png')
    plt.close()
    
    # Interaction type distribution
    plt.figure(figsize=(10, 6))
    df['interaction_type'].value_counts().plot(kind='bar')
    plt.title('Interaction Type Distribution')
    plt.xlabel('Interaction Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/interaction_types.png')
    plt.close()
    
    # Device type distribution
    plt.figure(figsize=(10, 6))
    df['device_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Device Type Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/device_types.png')
    plt.close()
    
    print(f"Session statistics plots saved to {output_dir}/")

def analyze_recommendations(recommender, test_sessions, top_k=3):
    """Analyze recommendation performance in detail."""
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
        recommendations = recommender.recommend_next(current_item, top_k=top_k)
        recommended_items = [item for item, _ in recommendations]
        
        # Update hit counts
        if actual_next in recommended_items:
            hits[current_item] += 1
        total_predictions[current_item] += 1
    
    # Calculate hit rates
    hit_rates = {}
    for item in total_predictions:
        hit_rates[item] = hits[item] / total_predictions[item]
    
    # Print analysis
    print("\nDetailed Recommendation Analysis:")
    print(f"Total test sessions: {len(test_sessions)}")
    print(f"Total predictions made: {sum(total_predictions.values())}")
    print(f"Overall hit rate: {sum(hits.values()) / sum(total_predictions.values()):.2%}")
    
    print("\nTop 5 most predictable items (highest hit rate):")
    sorted_items = sorted(hit_rates.items(), key=lambda x: x[1], reverse=True)
    for item, rate in sorted_items[:5]:
        print(f"{item}: {rate:.2%} ({hits[item]}/{total_predictions[item]})")
    
    print("\nTop 5 most popular items:")
    sorted_popular = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    for item, count in sorted_popular[:5]:
        print(f"{item}: {count} occurrences")

def main():
    # Load the data
    print("Loading session data...")
    df = load_sessions()
    
    # Prepare sessions for training
    print("\nPreparing sessions for training...")
    sessions = prepare_sessions_for_training(df)
    
    # Train the recommender
    print("\nTraining Markov Chain recommender...")
    recommender = MarkovChainRecommender()
    recommender.fit(sessions)
    
    # Create visualizations
    print("\nCreating visualizations...")
    import os
    os.makedirs('data', exist_ok=True)
    
    # Plot transition matrix
    plot_transition_matrix(recommender.get_transition_matrix())
    
    # Plot transition network
    plot_transition_network(recommender.get_transition_matrix())
    
    # Plot session statistics
    plot_session_statistics(df)
    
    # Analyze recommendations
    print("\nAnalyzing recommendations...")
    analyze_recommendations(recommender, sessions)

if __name__ == "__main__":
    main() 