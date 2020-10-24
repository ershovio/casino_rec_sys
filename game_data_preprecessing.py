import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, spmatrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_master_game(data_path: str) -> spmatrix:
    """
    Creates game-features matrix from `master_game_features` file
    :param data_path: path to the csv file
    :return: game-features matrix
    """
    df = pd.read_csv(data_path)
    df["jackpot"] = df["jackpot"].apply(lambda x: 1 if x == "YES" else 0)

    categorical_features = [
        "provider",
        "variance",
        "layout",
        "technology",
        "devices",
        "game_type"
    ]
    numerical_features = [
        "rtp",
        "max_bet",
        "betway",
        "max_coin",
        "slot_rank",
        "jackpot"
    ]
    need_scaled = [
        "rtp",
        "max_bet",
        "betway",
        "max_coin",
        "slot_rank"
    ]

    for c in categorical_features:
        df[c] = df[c].replace({np.nan: "None"})
    for c in numerical_features:
        df[c] = df[c].replace({np.nan: 0})

    encoder = OneHotEncoder()
    categorical_data = encoder.fit_transform(df[categorical_features])

    scaler = StandardScaler()
    df[need_scaled] = scaler.fit_transform(df[need_scaled])
    df[need_scaled] = df[need_scaled]

    number_of_rows = df["game_code"].max() + 1
    number_of_cat_features = categorical_data.shape[1]
    number_of_numerical_features = len(numerical_features)
    number_of_columns = number_of_cat_features + number_of_numerical_features
    item_feature_matrix = lil_matrix((number_of_rows, number_of_columns))

    for i, (ind, row) in enumerate(df.iterrows()):
        item_feature_matrix[ind, :number_of_cat_features] = categorical_data[i]
        item_feature_matrix[ind, number_of_cat_features:] = row[numerical_features]
    item_feature_matrix = item_feature_matrix.tocsr()
    return item_feature_matrix


def read_game_feature_derived(data_path: str) -> np.array:
    """
    Creates game-features matrix from `game_feature_derived` file
    :param data_path: path to the csv file
    :return: game-features matrix
    """
    df = pd.read_csv(data_path, parse_dates=["game_first_played"])
    df = df.replace({np.nan: 0})

    feature_columns = [
        "avg_daily_users",
        "total_users",
        "user_stickiness",
        "n_days_played",
        "avg_daily_plays",
        "total_plays",
        "days_available",
        "stickiness_trending",
        "stickiness_percentile",
        "popularity_trending",
        "popularity_percentile",
        "new_to_game_stickiness",
        "new_user_stickiness"
    ]

    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    number_of_rows = df["gamecode"].max() + 1
    number_of_columns = len(feature_columns)
    matrix = np.zeros(shape=(number_of_rows, number_of_columns))

    for ind, row in df.iterrows():
        matrix[ind] = row[feature_columns]
    return matrix
