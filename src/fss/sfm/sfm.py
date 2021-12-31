import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel

def sfm_coef_rank(estimator,X_train, y_train):
    selector = SelectFromModel(estimator)
    selector = selector.fit(X_train, y_train)
    score = np.abs(selector.estimator_.coef_[0])
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": score})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score', ascending=False).reset_index(drop=True)
    return ranked_features


def sfm_fi_rank(estimator,X_train, y_train):
    selector = SelectFromModel(estimator)
    selector = selector.fit(X_train, y_train)
    score = np.abs(selector.estimator_.feature_importances_)
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": score})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score', ascending=False).reset_index(drop=True)
    return ranked_features