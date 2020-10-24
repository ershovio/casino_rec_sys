import logging
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config
from game_data_preprecessing import read_master_game, read_game_feature_derived
from metrics import precision_at_k
from recommenders import CosineRecommender, HybridRecommender, MemoryBasedRecommender
from user_data_preprocessing import (
    read_users_data,
    split_users_on_train_test,
    create_test_dateset,
    create_user_item_matrix
)


def prepare_test_train_df(conf: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grouped_users = read_users_data(conf.user_plays_path_path)
    logging.info("Users data is loaded")
    train_df, test_df = split_users_on_train_test(
        df=grouped_users,
        test_ratio=conf.test_train_ratio,
        lower_bound=conf.lower_bound,
        upper_bound=conf.upper_bound
    )
    logging.info(f"Train and test datasets are created. "
                 f"Train shape is {train_df.shape}, test shape is {test_df.shape}")
    return train_df, test_df


def train_collaborative_filtering_model(
        train_df: pd.DataFrame
) -> Tuple[CosineRecommender, StandardScaler]:
    user_item_matrix, scaler = create_user_item_matrix(train_df)
    logging.info(f"User matrix item shape is {user_item_matrix.shape}")
    collaborative_filtering_recommender = CosineRecommender()
    collaborative_filtering_recommender.fit(user_item_matrix.T)
    logging.info("Collaborative filtering recommender is trained")
    return collaborative_filtering_recommender, scaler


def train_content_based_master_game_model(conf: Config) -> CosineRecommender:
    master_game_matrix = read_master_game(conf.master_game_features_path)
    logging.info(f"Master game matrix shape is {master_game_matrix.shape}")
    content_based_master_game_recommender = CosineRecommender()
    content_based_master_game_recommender.fit(master_game_matrix)
    logging.info("Content based recommender on master game data is trained")
    return content_based_master_game_recommender


def train_content_based_feature_derived_model(conf: Config) -> CosineRecommender:
    game_feature_derived_matrix = read_game_feature_derived(conf.game_feature_derived_path)
    logging.info(f"Game feature derived matrix shape is {game_feature_derived_matrix.shape}")
    content_based_feature_derived_recommender = CosineRecommender()
    content_based_feature_derived_recommender.fit(game_feature_derived_matrix)
    logging.info("Content based recommender on feature derived data is trained")
    return content_based_feature_derived_recommender


def evaluate_results(
        conf: Config,
        recommender: MemoryBasedRecommender,
        scaler: StandardScaler,
        test_df: pd.DataFrame
) -> float:
    number_of_items = recommender.number_of_items()
    users, labels = create_test_dateset(
        test_df,
        scaler=scaler,
        n_items=number_of_items,
        fl_ratio=conf.feature_label_ratio
    )
    predicted = recommender.predict(users)
    metric = precision_at_k(labels, predicted, k=conf.precision_at_k)
    logging.info(f"Precision@{conf.precision_at_k} on test dataset: {metric:0.3f}")
    return metric


def train_pipeline(conf: Config):
    """
    Performs that whole training pipeline:
    * reads and preprocesses training data
    * fits 3 recommenders
    * computes test metrics using hybrid weighted model
    """
    # read users data
    train_df, test_df = prepare_test_train_df(conf)

    # train recommenders
    collaborative_filtering_recommender, scaler = train_collaborative_filtering_model(train_df)
    content_based_master_game_recommender = train_content_based_master_game_model(conf)
    content_based_feature_derived_recommender = train_content_based_feature_derived_model(conf)

    # combine all recommenders using weighted hybrid model
    recommenders = [
        collaborative_filtering_recommender,
        content_based_master_game_recommender,
        content_based_feature_derived_recommender
    ]
    coefficients = [
        conf.collaborative_filtering_coef,
        conf.cb_master_game_coef,
        conf.cb_feature_derived_coef
    ]
    hybrid_recommender = HybridRecommender(recommenders, coefficients)

    # evaluate results
    metric = evaluate_results(conf, hybrid_recommender, scaler, test_df)
    return hybrid_recommender, metric


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    conf = Config.load_from_file()
    recommender, metric = train_pipeline(conf)
