import numpy as np
import pandas as pd
import os
from time import perf_counter
from skrebate import ReliefF
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis


def relief_ranking(X_train, y_train):
	X_train_values, y_train_values = X_train.values, y_train
	selector = ReliefF(n_features_to_select=X_train.shape[1], n_neighbors=0.01, n_jobs=12)
	selector.fit(X_train_values, y_train_values)
	rank_df = pd.DataFrame({"sub_feature" : X_train.columns[selector.top_features_], "rank":list(range(1, 1+X_train.shape[1]))})
	rank_df['feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
	ranked_features = rank_df.groupby('feature',as_index=False).agg('mean').sort_values('rank').reset_index(drop=True)
	ranked_features['Score'] = 1/ranked_features['rank']
	ranked_features.drop('rank',axis=1,inplace=True)
	return ranked_features