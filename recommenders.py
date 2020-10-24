import math
from abc import ABC, abstractmethod
from typing import Union, List

import numpy as np
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity


class MemoryBasedRecommender(ABC):
    """
    Abstract class for recommenders that use weights matrices to compute similarity between items
    """

    @abstractmethod
    def predict_proba(self, x: Union[np.array, spmatrix]) -> np.array:
        """
        Computes weights for each feature vector
        :param x: feature vector
        :return: weight for each feature vector
        """

    @abstractmethod
    def number_of_items(self) -> int:
        """
        :return: Number of rows (columns) in weights matrix
        """

    def predict(self, x: Union[np.array, spmatrix]) -> np.array:
        """
        Ranks items for each vector in x
        :param x: feature vector
        :return: Ranked items for each vector in x
        """
        predictions = self.predict_proba(x)
        return np.argsort(predictions)[:, ::-1]


class CosineRecommender(MemoryBasedRecommender):
    """
    Implements memory-based recommendations using cosine similarity
    """

    _NOT_FIT_WARNING = "Weights matrix must be defined, invoke `fit` method"
    _is_fit = False
    _weights_matrix: np.array

    def __init__(self, weights_matrix: np.array = None):
        """
        :param weights_matrix: item weights matrix
        """
        if weights_matrix is not None:
            self._weights_matrix = weights_matrix
            self._is_fit = True

    def fit(self, m: Union[spmatrix, np.array]):
        """
        Computes item weights matrix
        :param m: item matrix
        """
        if isinstance(m, spmatrix):
            m = m.tocsr()
        self._weights_matrix = cosine_similarity(m, m)
        self._is_fit = True

    def predict_proba(self, x: Union[np.array, spmatrix]) -> np.array:
        """
        Computes weights for each feature vector
        :param x: feature vector
        :return: weight for each feature vector
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        weights = x.dot(self._weights_matrix)
        return weights

    def save(self, file_path: str):
        """
        Saves recommender to a file
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        np.save(self._weights_matrix, file_path)

    def number_of_items(self) -> int:
        """
        :return: Number of rows (columns) in weights matrix
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        return self._weights_matrix.shape[0]

    @staticmethod
    def load_from_file(file_path: str) -> "CosineRecommender":
        """
        Loads recommender from a file
        """
        item_matrix = np.load(file_path)
        return CosineRecommender(item_matrix)


class HybridRecommender(MemoryBasedRecommender):
    """
    Implements weighted hybrid model, e.g.
    weight(x) = sum(c_i * r_i(x)), where r_i and c_i are inner recommenders and coefficients
    """

    def __init__(
            self,
            recommenders: List[MemoryBasedRecommender],
            coefficients: Union[List[float], np.array]
    ):
        assert math.isclose(sum(coefficients), 1, abs_tol=1e-6)
        assert len([c for c in coefficients if c < 0]) == 0
        assert len(recommenders) == len(coefficients)
        num_of_items = recommenders[0].number_of_items()
        for r in recommenders:
            assert r.number_of_items() == num_of_items, \
                ", ".join([str(i.number_of_items()) for i in recommenders])
        self.recommenders = recommenders
        self.coefficients = coefficients

    def predict_proba(self, x: Union[np.array, spmatrix]) -> np.array:
        predictions = np.array([
            r.predict_proba(x) * c for r, c in zip(self.recommenders, self.coefficients)
        ])
        predictions = np.sum(predictions, axis=0)
        return predictions

    def number_of_items(self):
        """
        :return: Number of rows (columns) in weights matrix of each inner recommender
        """
        return self.recommenders[0].number_of_items()
