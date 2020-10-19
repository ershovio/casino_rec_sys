import numpy as np


def precision_at_k(relevant_items: np.array, predicted_items: np.array, k: int = 5) -> float:
    """
    Computes precision@k metric
    """
    assert relevant_items.shape[0] == predicted_items.shape[0]
    res = []
    for ri, pi in zip(relevant_items, predicted_items):
        assert len(relevant_items) > 0
        intersections = np.in1d(pi, ri).astype(int)
        pre_at_k = sum(intersections[:k]) / min(len(ri), k)
        res.append(pre_at_k)
    return np.mean(res)
