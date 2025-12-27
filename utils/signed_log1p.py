import numpy as np
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
        return signed_log1p_func(np.array(X))