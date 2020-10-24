import yaml


class Config:
    def __init__(self, data: dict):
        self.game_feature_derived_path = data["data_paths"]["game_feature_derived"]
        self.master_game_features_path = data["data_paths"]["master_game_features"]
        self.user_plays_path_path = data["data_paths"]["user_plays_path"]
        self.game_feature_derived_coef = data["coefficients"]["game_feature_derived"]
        self.master_game_features_coef = data["coefficients"]["master_game_features"]
        self.user_plays_coef = data["coefficients"]["user_plays"]
        self.precision_at_k = data["test"]["precision_at_k"]
        self.lower_bound = data["test"]["lower_bound"]
        self.upper_bound = data["test"]["upper_bound"]
        self.test_train_ratio = data["test"]["test_train_ratio"]
        self.feature_label_ratio = data["test"]["feature_label_ratio"]

    @staticmethod
    def load_from_file() -> "Config":
        """
        Reads config from the default YAML file
        """
        with open("config.yml") as f:
            data = yaml.safe_load(f)
        return Config(data)
