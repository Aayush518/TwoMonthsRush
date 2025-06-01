import pytest
from src.data.synthetic_session_generator import SyntheticSessionGenerator

def test_generate_sessions_length():
    items = ['A', 'B', 'C']
    gen = SyntheticSessionGenerator(items, num_users=5, min_session_length=2, max_session_length=4)
    sessions = gen.generate_sessions()
    assert len(sessions) == 5
    for session in sessions:
        assert 2 <= len(session) <= 4
        for item in session:
            assert item in items
