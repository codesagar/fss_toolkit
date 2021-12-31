import pandas as pd
from sklearn.feature_selection import SelectKBest

def skb_rank(estimator,X_train, y_train):
    selector = SelectKBest(score_func=estimator,k='all').fit(X_train,y_train)
    rank_df = pd.DataFrame({"sub_feature":X_train.columns, "Score": selector.scores_})
    rank_df['Feature'] = rank_df['sub_feature'].apply(lambda x: "_".join(x.split("_")[:-1]))
    ranked_features = rank_df.groupby('Feature',as_index=False).agg('mean').sort_values('Score',ascending=False).reset_index(drop=True)
    return ranked_features