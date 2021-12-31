import numpy as np
import pandas as pd
import os
from time import perf_counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis
from mrmr import mrmr_classif

  
def mrmr_ranking(X_train, y_train):
    selected_features = mrmr_classif(X_train, y_train, K = X_train.shape[1])
    rank_df = pd.DataFrame({"sub_feature" : selected_features, "Rank":list(range(1, 1+len(selected_features)))})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Rank').reset_index(drop=True)
    ranked_features['Score'] = 1/ranked_features['Rank']
    ranked_features.drop('Rank',axis=1,inplace=True)
    return ranked_features
