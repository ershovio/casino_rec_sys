from collections import Counter
from typing import Union

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, spmatrix
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    """
    Implements item-based memory-based collaborative filtering
    using cosine similarity
    """
    _NOT_FIT_WARNING = "Item matrix must be defined, invoke `fit` method"
    _is_fit = False
    _item_matrix: np.array

    def __init__(self, item_matrix: np.array = None):
        """
        :param item_matrix: item weights matrix
        """
        if item_matrix is not None:
            self._item_matrix = item_matrix
            self._is_fit = True

    def fit(self, user_item_matrix: spmatrix):
        """
        Computes item weights matrix
        :param user_item_matrix:
        """
        a = user_item_matrix.T.tocsr()
        self._item_matrix = cosine_similarity(a, a)
        self._is_fit = True

    def predict(self, users: Union[np.array, spmatrix]) -> np.array:
        """
        Ranks each item for each user in `users`
        :param users: user-item matrix (users[i, j] = number of events j for user i)
        :return: ranked items for each user
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        weights = users.dot(self._item_matrix)
        predictions = np.argsort(weights)[:, ::-1]
        return predictions

    def save(self, file_path: str):
        """
        Saves recommender to a file
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        np.save(self._item_matrix, file_path)

    def number_of_items(self) -> int:
        """
        :return: Number of rows (columns) in weights matrix
        """
        assert self._is_fit, self._NOT_FIT_WARNING
        return self._item_matrix.shape[0]

    @staticmethod
    def load_from_file(file_path: str) -> "CollaborativeFilteringRecommender":
        """
        Loads recommender from a file
        """
        item_matrix = np.load(file_path)
        return CollaborativeFilteringRecommender(item_matrix)

    @staticmethod
    def create_from_dataframe(grouped_users: pd.DataFrame) -> "CollaborativeFilteringRecommender":
        """
        Creates recommender and fits it using given dataframe
        :param grouped_users: users dataframe
        """
        user_item_matrix = CollaborativeFilteringRecommender._create_user_item_matrix(grouped_users)
        rec = CollaborativeFilteringRecommender()
        rec.fit(user_item_matrix)
        return rec

    @staticmethod
    def _create_user_item_matrix(grouped_users: pd.DataFrame) -> spmatrix:
        number_of_rows = grouped_users.index.max() + 1
        number_of_columns = grouped_users["gamecode"].apply(lambda x: max(x)).max() + 1
        user_item_matrix = lil_matrix((number_of_rows, number_of_columns))
        for i, row in grouped_users.iterrows():
            cnt = Counter(row["gamecode"])
            for k, v in cnt.items():
                user_item_matrix[i, k] = v
        return user_item_matrix
