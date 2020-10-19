from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, spmatrix


def read_users_data(data_path: str) -> pd.DataFrame:
    """
    Reads data from users csv file that contains
    `userid`, `gamecode`, and `event_datetime` columns
    and groups it by userid
    :param data_path: path to the file
    """
    df = pd.read_csv(data_path)
    df = df[df["gamecode"].notnull()]
    df["gamecode"] = df["gamecode"].astype(int)
    df['event_datetime'] = pd.to_datetime(df['event_datetime'])
    df["date"] = df["event_datetime"].apply(lambda x: x.date())
    df["time"] = df["event_datetime"].apply(lambda x: x.time())
    grouped_df = df.groupby("userid").agg(list)
    grouped_df["number_of_events"] = grouped_df["gamecode"].apply(lambda x: len(x))
    return grouped_df


def split_users_on_train_test(
        df: pd.DataFrame,
        test_ratio: float = 0.05,
        lower_bound: int = 3,
        upper_bound: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split users dataframe (created by `read_users_data` method) on train and test parts
    :param df: dataframe to split
    :param test_ratio: what ratio the test part will have
    :param lower_bound: min number of events for a test user
    :param upper_bound: max number of events for a test user
    :return: train, test dataframes
    """
    test_candidates = df[(df["number_of_events"] > lower_bound) &
                         (df["number_of_events"] < upper_bound)]
    test_samples_df = test_candidates.sample(
        frac=test_ratio * df.shape[0] / test_candidates.shape[0]
    )
    test_samples_ids = test_samples_df.index.to_numpy()
    test_users = np.in1d(df.index, test_samples_ids)
    train_users = np.logical_not(test_users)
    return df.loc[train_users], df.loc[test_users]


def create_test_dateset(
        df: pd.DataFrame,
        n_items: int,
        fl_ratio: float = 0.75
) -> Tuple[spmatrix, pd.Series]:
    """
    Creates test data from a dataframe in this way:
    a user's events are sorted by timestamp and then
    the first part is treated as "features" and the last part is treated as "labels"
    :param df: users dataframe (created by `read_users_data` method)
    :param n_items: number of items in collaborative filtering weights matrix
    :param fl_ratio: what ratio the "feature" part will have
    :return:  tuple of (features, labels)
    """
    df = df.copy(deep=True)
    df["events_with_time"] = list(zip(df["gamecode"], df["event_datetime"]))
    df["dt_sorted"] = df["event_datetime"].apply(lambda x: np.argsort(x))
    df["events_time_sorted"] = df.apply(lambda r: np.array(r["gamecode"])[r["dt_sorted"]], axis=1)
    df["test_events"] = df["events_time_sorted"].apply(lambda x: x[:int(fl_ratio * x.shape[0])])
    df["label"] = df["events_time_sorted"].apply(lambda x: x[int(fl_ratio * x.shape[0]):])
    number_of_rows = df.shape[0]
    number_of_columns = n_items
    matrix = lil_matrix((number_of_rows, number_of_columns))
    for ind, (i, row) in enumerate(df.iterrows()):
        cnt = Counter(row["test_events"])
        for k, v in cnt.items():
            matrix[ind, k] = v
    return matrix, df["label"]


def create_small_users_dataset(
        data_path: str = "data/user_plays_enc.csv",
        new_path: str = "data/user_plays_enc_small.csv",
        number_of_events: int = 50000
):
    """
    Creates small users subset to test the whole logic locally
    :param data_path: path to the original csv file
    :param new_path: path to the new csv file
    :param number_of_events: number of events in the new file
    """
    df = pd.read_csv(data_path)
    df.iloc[:number_of_events].to_csv(new_path)
