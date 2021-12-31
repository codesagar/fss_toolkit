import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from time import perf_counter


def rfe_rank(estimator, X_train, y_train):
		selector = RFE(estimator, n_features_to_select=1, step=2)   
		selector.fit(X_train, y_train)
		rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Rank": selector.ranking_})
		rank_df['feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
		ranked_features = rank_df.groupby('feature',as_index=False).agg('mean').sort_values('Rank').reset_index(drop=True)
		ranked_features['Score'] = 1/ranked_features['Rank']
		ranked_features.drop('Rank',axis=1,inplace=True)
		return ranked_features
