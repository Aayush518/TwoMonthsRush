from collections import defaultdict
from typing import List, Dict, Tuple

class MarkovChainRecommender:
    """
    Builds a Markov chain transition matrix from session data and provides item-to-item recommendations.
    """
    def __init__(self):
        self.transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_matrix: Dict[str, Dict[str, float]] = {}

    def fit(self, sessions: List[List[str]]) -> None:
        for session in sessions:
            for i in range(len(session) - 1):
                curr_item, next_item = session[i], session[i + 1]
                self.transition_counts[curr_item][next_item] += 1
        self.transition_matrix = {}
        for curr_item, next_items in self.transition_counts.items():
            total = sum(next_items.values())
            self.transition_matrix[curr_item] = {k: v / total for k, v in next_items.items()}

    def recommend_next(self, current_item: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if current_item not in self.transition_matrix:
            return []
        sorted_recs = sorted(
            self.transition_matrix[current_item].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_recs[:top_k]

    def get_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        return self.transition_matrix
