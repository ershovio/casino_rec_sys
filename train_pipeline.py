import logging

from collaborative_filtering import CollaborativeFilteringRecommender
from data_preprocessing import read_users_data, split_users_on_train_test, create_test_dateset
from metrics import precision_at_k


def train_pipeline(
        users_data: str = "data/user_plays_enc_small.csv"
):
    """
    Performs that whole training pipeline:
    * reads and preprocesses training data
    * fits collaborative filtering model
    * computes test metrics
    :param users_data: path to users csv file
    """
    grouped_users = read_users_data(users_data)
    logging.info("Users data is loaded")
    train_df, test_df = split_users_on_train_test(grouped_users)
    logging.info("Train and test datasets are created")
    recommender = CollaborativeFilteringRecommender.create_from_dataframe(train_df)
    logging.info("Collaborative filtering recommender is trained")
    number_of_items = recommender._item_matrix.shape[0]
    users, labels = create_test_dateset(test_df, number_of_items)
    predicted = recommender.predict(users)
    metric = precision_at_k(labels, predicted, k=5)
    logging.info(f"Precision@5 on test dataset: {metric:0.3f}")
    return recommender, metric


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    recommender, metric = train_pipeline()
