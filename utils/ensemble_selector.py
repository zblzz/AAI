from functools import partial
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scipy.stats import rankdata, spearmanr
import numpy as np

def rank01(scores):
    """
    将分数转换为 0~1 的排名 (处理平局，越大越好)
    """
    scores = np.asarray(scores)
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=float)
    # method='average' 确保平局获得平均排名
    ranks = rankdata(scores, method='average')
    return (ranks - 1) / (len(scores) - 1)

class EnsembleSelector(BaseEstimator, TransformerMixin):
    """
    特征选择器 (EnsembleSelector)
    """
    def __init__(self, n_features_to_select=2000, 
                 l1_C=1.0,               # [修复] 添加参数
                 mi_n_neighbors=3,       # [修复] 添加参数
                 verbose=False, seed=42):
        self.n_features_to_select = n_features_to_select
        self.l1_C = l1_C
        self.mi_n_neighbors = mi_n_neighbors
        self.verbose = verbose
        self.seed = seed
        self.selected_indices_ = None
        
        # XGB 参数 (保持笨拙模式以防过拟合)
        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_jobs': 1,
            'eval_metric': 'logloss',
            'random_state': seed
        }

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_total = X.shape[1]
        
        if self.verbose:
            print(f"   [Ensemble] Scoring {n_total} features...")
            print(f"     -> Params: l1_C={self.l1_C}, mi_k={self.mi_n_neighbors}")

        # --- Judge 1: Mutual Information ---
        # MI 计算是单线程的，不受 n_jobs 影响
        mi_scorer = partial(mutual_info_classif, random_state=self.seed, n_neighbors=self.mi_n_neighbors)
        score_mi = rank01(mi_scorer(X, y))
        
        # --- Judge 2: Lasso ---
        clf_l1 = LogisticRegression(penalty='l1', C=self.l1_C, solver='liblinear', 
                                    class_weight='balanced', random_state=self.seed)
        clf_l1.fit(X, y)
        score_lasso = rank01(np.abs(clf_l1.coef_[0]))
        
        # --- Judge 3: XGBoost ---
        # 【关键修正】n_jobs=1，防止与外层 CV 抢资源导致死锁或抖动
        clf_xgb = XGBClassifier(**self.xgb_params)
        clf_xgb.fit(X, y)
        score_xgb = rank01(clf_xgb.feature_importances_)
        
        # --- Weighted Sum ---
        final_score = (0.4 * score_lasso) + (0.4 * score_xgb) + (0.2 * score_mi)
        
        # --- Top K ---
        k = min(self.n_features_to_select, n_total)
        self.selected_indices_ = np.argsort(final_score)[::-1][:k]
        
        if self.verbose:
            idx = self.selected_indices_
            avg_lasso = np.mean(score_lasso[idx])
            avg_xgb = np.mean(score_xgb[idx])
            avg_mi = np.mean(score_mi[idx])
            print(f"   [Ensemble] Top {k} selected. Cutoff: {final_score[idx[-1]]:.4f}")
            print(f"     -> Support: Lasso={avg_lasso:.2f}, XGB={avg_xgb:.2f}, MI={avg_mi:.2f}")

        return self

    def transform(self, X):
        X = np.array(X)
        return X[:, self.selected_indices_]