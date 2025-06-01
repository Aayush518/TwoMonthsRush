from src.recommender.markov_chain_recommender import MarkovChainRecommender

def test_fit_and_recommend():
    sessions = [['A', 'B', 'C'], ['A', 'C', 'B'], ['B', 'A', 'C']]
    rec = MarkovChainRecommender()
    rec.fit(sessions)
    matrix = rec.get_transition_matrix()
    assert 'A' in matrix
    assert isinstance(matrix['A'], dict)
    recs = rec.recommend_next('A', top_k=2)
    assert isinstance(recs, list)
    assert all(isinstance(x, tuple) for x in recs)
