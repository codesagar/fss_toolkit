# %% [markdown]
# ## Loading data
# - 735 datapoints each having 24 features for 60 timesteps
# - data.shape => (735,60,24)  => (datapoints, timesteps, features)

# %%
import pandas as pd
import numpy as np
data = np.load("sample_data.npz")

# %% [markdown]
# ### FSS on TS data (3 dimentions)

# %% [markdown]
# ### FCBF - Fast Correlation Based Filter 

# %%
from src.fss.fcbf.fcbf import FCBF
fcbf = FCBF(data=data)    ## data is dictionary with 2 keys "np_data" having 3d numpy data and "subclass" corresponding to target classes
fcbf_rank = fcbf.rank()
fcbf_rank.head()

# %% [markdown]
# ## CSFS

# %%
# from src.fss.csfs.csfs import CSFS
# csfs = CSFS(data=data)    ## data is dictionary with 2 keys "np_data" having 3d numpy data and "subclass" corresponding to target classes
# csfs_rank = csfs.rank()
# csfs_rank.head()

# %% [markdown]
# ## Vectorizing data
# - (735,60,24) -> (735,168)
# - where 168 corresponds represents 24 features with 7 statistical features each

# %%
from src.preprocessing.vectorize import vectorize
vectorized_data = vectorize(data['np_data'])
print(vectorized_data.shape)
vectorized_data.head()

# %% [markdown]
# ## Binarizing target

# %%
y_train_bin = np.where(data['target']=='NF',0,1)

# %% [markdown]
# ### MRMR - Maximum Relevance Minimum Redundancy

# %%
from src.fss.mrmr.mrmr import mrmr_ranking
mrmr_rank = mrmr_ranking(vectorized_data, y_train_bin)
mrmr_rank.head()


# %% [markdown]
# ### RelieF

# %%
from src.fss.relief.relief import relief_ranking
relief_rank = relief_ranking(vectorized_data, y_train_bin)
relief_rank.head()


# %% [markdown]
# ### Recursive Feature Elimination with Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from src.fss.rfe.rfe import rfe_rank


logistic = LogisticRegression(solver='liblinear',random_state=777)
rfe_logistic_ranks = rfe_rank(logistic, vectorized_data, y_train_bin)
rfe_logistic_ranks.head()

# %% [markdown]
# ### Select From Model with RandomForest

# %%
from src.fss.sfm.sfm import sfm_fi_rank
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=50)
sfm_rf_ranks = sfm_fi_rank(rf, vectorized_data, y_train_bin)
sfm_rf_ranks.head()


# %% [markdown]
# ### Select K Best with MutualInfo

# %%
from src.fss.skb.skb import skb_rank
from sklearn.feature_selection import mutual_info_classif

skb_mi_ranks = skb_rank(mutual_info_classif, vectorized_data, y_train_bin)
skb_mi_ranks.head()

# %%



