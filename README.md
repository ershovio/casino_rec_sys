# Recommender system for casino

## Algorithm
The system uses hybrid weighted model that consist of 3 inner algorithms.

<img src="https://render.githubusercontent.com/render/math?math=r=\sum_{i=1}^3{c_i*r_i(x)}">

where `ri` is inner recommender and `ci` is corresponding coefficient. 
This logic is implemented in [HybridRecommender](recommenders.py#L99).

All recommenders are memory-based. 
They use cosine-similarity matrix. 
All of them use [CosineRecommender](recommenders.py#L39) that 
implements predictions using cosine-similarity matrix.

### Collaborative-filtering recommender
It uses user-item matrix (where `m[u, g]=k` means that user `u` played in game `g` `k` times) 
to compute item-item similarity matrix. 
Data preprocessing for this part can be found in [user_data_preprocessing.py](user_data_preprocessing.py)

### Content-based recommenders
There are 2 content-based recommenders. The first is based on data from `master_game_features.csv` 
and the second is based on data from `game_feature_derived_enc.csv`. 
They have pretty similar logic when data from corresponding csv files are used 
to compute item-item similarity matrix.
Data preprocessing for this part can be found in [game_data_preprecessing.py](game_data_preprecessing.py)

## Training
The whole training pipeline can be found in [train_pipeline](train_pipeline.py).

Example of searching coefficients for the hybrid model can be found in [train_eval](train_eval.ipynb).

## Evaluation

### Test dataset creation
* [Split users on test and train parts](user_data_preprocessing.py#L28)
* For each user in the test part treat `n` oldest events 
as "features" and the rest events as "labels"
* Create user-item matrix using "feature" events

### Metrics calculation
* Compute prediction using user-item matrix from the previous part
* Computer `Precision@K` metric using "labels" part from the previous part
