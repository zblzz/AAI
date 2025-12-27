import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. 核心魔法：Signed Log1p 变换 (来自 0.77 Baseline)
# ==========================================
def signed_log1p_func(X):
    # 保持符号的同时做 log1p，把长尾压扁，把正负号留住
    return np.sign(X) * np.log1p(np.abs(X))

class SignedLog1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            arr = np.sign(X.values) * np.log1p(np.abs(X.values))
            return pd.DataFrame(arr, columns=X.columns, index=X.index)
        arr = np.asarray(X)
        return np.sign(arr) * np.log1p(np.abs(arr))