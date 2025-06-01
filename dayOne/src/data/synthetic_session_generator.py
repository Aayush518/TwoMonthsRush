import random
from typing import List

class SyntheticSessionGenerator:
    """
    Generates synthetic user session data for recommendation system experiments.
    """
    def __init__(self, items: List[str], num_users: int = 100, min_session_length: int = 5, max_session_length: int = 10, neighbor_bias: float = 0.7):
        self.items = items
        self.num_users = num_users
        self.min_session_length = min_session_length
        self.max_session_length = max_session_length
        self.neighbor_bias = neighbor_bias

    def generate_sessions(self) -> List[List[str]]:
        sessions = []
        for _ in range(self.num_users):
            session_length = random.randint(self.min_session_length, self.max_session_length)
            session = [random.choice(self.items)]
            for _ in range(session_length - 1):
                if random.random() < self.neighbor_bias:
                    current_index = self.items.index(session[-1])
                    neighbors = [
                        self.items[(current_index + 1) % len(self.items)],
                        self.items[(current_index - 1) % len(self.items)]
                    ]
                    next_item = random.choice(neighbors)
                else:
                    next_item = random.choice(self.items)
                session.append(next_item)
            sessions.append(session)
        return sessions
