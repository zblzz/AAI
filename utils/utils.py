import os
import numpy as np
import pandas as pd
import joblib
from scipy.linalg import sqrtm, inv
from utils.ensemble_selector import EnsembleSelector
from utils.signed_log1p import SignedLog1pTransformer

# Sklearn Imports
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import StackingClassifier

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import umap

def get_dim_reducer(method_name, n_components, seed=42, **kwargs):
    """
    获取降维/特征选择器
    """
    # 获取参数
    spca_k = kwargs.get('spca_k', 3000)
    k_val = 'all' if str(spca_k) == 'all' else int(spca_k)
    
    # 获取 EnsembleSelector 的特定参数
    l1_c = float(kwargs.get('ens_l1_c', 1.0))
    mi_neighbors = int(kwargs.get('ens_mi_k', 3))
    verbose = bool(kwargs.get('ens_verbose', False))

    # === 基础选择器 ===
    base_selector = SelectKBest(score_func=f_classif, k=k_val)

    if method_name == 'pca':
        return PCA(n_components=n_components, random_state=seed)
    
    elif method_name == 'spca': 
        # 经典方案：Linear Selection -> Linear PCA
        pca = PCA(n_components=n_components, random_state=seed)
        return Pipeline([('selector', base_selector), ('pca', pca)])
    
    elif method_name == 'baseline_select':
        # 【偷师点 2】Baseline 的纯 F-test
        print(f"   [Reducer] Using Baseline SelectKBest(f_classif, k={k_val})")
        return SelectKBest(score_func=f_classif, k=k_val)
    
    elif method_name == 'umap':
        try:
            import umap
        except ImportError:
            raise ImportError("Please install umap: pip install umap-learn")
        
        reducer = umap.UMAP(n_components=n_components, 
                            n_neighbors=15, 
                            min_dist=0.1, 
                            metric='euclidean',
                            target_metric='categorical', 
                            random_state=seed,
                            n_jobs=1)
        return Pipeline([('selector', base_selector), ('umap', reducer)])
    
    elif method_name == 'poly_pca':
        poly_k = 50 
        selector_strict = SelectKBest(score_func=f_classif, k=poly_k)
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        pca = PCA(n_components=n_components, random_state=seed)
        return Pipeline([('selector', selector_strict), ('poly', poly), ('pca', pca)])
    
    elif method_name == 'fa':
        fa = FactorAnalysis(n_components=n_components, random_state=seed)
        return Pipeline([('selector', base_selector), ('fa', fa)])
    
    elif method_name == 'ensemble_spca':
        # 【关键】透传参数给 EnsembleSelector
        ensemble_sel = EnsembleSelector(
            n_features_to_select=k_val, 
            l1_C=l1_c,
            mi_n_neighbors=mi_neighbors,
            seed=seed,
            verbose=verbose
        )
        pca = PCA(n_components=n_components, random_state=seed)
        return Pipeline([('ensemble_selector', ensemble_sel), ('pca', pca)])
    
    elif method_name == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown method: {method_name}")

def get_classifier(method_name, seed=42, **kwargs):
    """
    获取分类器
    """
    if method_name == 'lr':
        return LogisticRegression(penalty='l2', C=0.1, solver='liblinear', 
                                  class_weight='balanced', random_state=seed)
    
    elif method_name == 'rf':
        return RandomForestClassifier(n_estimators=200, max_depth=7, 
                                      class_weight='balanced', random_state=seed)
    
    elif method_name == 'xgb':
        # 使用较稳健的参数，防止小样本过拟合
        return XGBClassifier(
            n_estimators=200, 
            max_depth=3, 
            learning_rate=0.05,
            subsample=0.7, 
            colsample_bytree=0.5,
            reg_alpha=2, 
            reg_lambda=3, 
            eval_metric='logloss',
            random_state=seed, 
            n_jobs=1
        )
        # return XGBClassifier(
        #     n_estimators=100,       # [砍] 减少树的数量，防止过拟合
        #     max_depth=2,            # [砍] 深度降为2，甚至可以试1。越浅越防过拟合
        #     learning_rate=0.03,     # [慢] 学习率再低一点
        #     subsample=0.6,          # [抖] 增加随机性
        #     colsample_bytree=0.4,   # [抖] 每次只看 40% 的特征
        #     reg_alpha=5,            # [狠] L1 正则直接拉满 (之前是2)，强迫稀疏
        #     reg_lambda=5,           # [狠] L2 正则也拉满
        #     eval_metric='logloss',
        #     random_state=seed, 
        #     n_jobs=-1
        # )
    
    elif method_name == 'svm_calib':
        base_svm = LinearSVC(
            C=0.005, 
            penalty='l2', 
            dual=True, 
            class_weight='balanced', 
            random_state=seed, 
            max_iter=5000
        )
        # base_svm = LinearSVC(
        #     C=0.01,            # [微调] 再强一点正则，试试 0.005
        #     penalty='l2', 
        #     loss='hinge',       # [关键修改] 使用标准的 Hinge Loss 而不是 squared_hinge
        #     dual=True,          # Hinge Loss 必须配 dual=True
        #     class_weight='balanced', 
        #     random_state=seed, 
        #     max_iter=10000
        # )
        return CalibratedClassifierCV(base_svm, method='sigmoid', cv=5)
    
    elif method_name == 'ridge':
        base_ridge = RidgeClassifier(class_weight='balanced', random_state=seed)
        return CalibratedClassifierCV(base_ridge, method='sigmoid', cv=3)
    
    elif method_name == 'gnb':
        return GaussianNB()
    
    elif method_name == 'lr_elastic':
        print(f"   [Classifier] LogisticRegression (ElasticNet, Saga)")
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.7,
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            n_jobs=1,
            random_state=seed
        )
    elif method_name == 'knn':
        # n_neighbors=7: 稍微大一点，增加鲁棒性（一般取 3-9）
        # weights='distance': 距离近的样本权重更大，比 uniform 更适合
        # metric='euclidean': 因为你做了 PCA，特征已经去相关了，欧氏距离有效
        return KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=1)
    elif method_name == 'voting':
        voters_str = kwargs.pop('voters', 'lr,svm_calib,xgb')
        voters_list = [v.strip() for v in voters_str.split(',')]
        weights = kwargs.pop('voting_weights', None)
        
        estimators = []
        for v_name in voters_list:
            clf = get_classifier(v_name, seed, **kwargs)
            estimators.append((v_name, clf))
            
        print(f"   [Voting] Voters: {voters_list} | Weights: {weights}")
        return VotingClassifier(estimators=estimators, voting='soft', weights=weights, n_jobs=1)
    
    elif method_name == 'stacking':
        # Stacking 需要保留 voters 参数
        voters_str = kwargs.get('voters', 'lr,svm_calib,xgb')
        voters_list = [v.strip() for v in voters_str.split(',')]
        
        print(f"   [Stacking] Base Learners: {voters_list}")
        
        estimators = []
        for v_name in voters_list:
            clf = get_classifier(v_name, seed, **kwargs)
            estimators.append((v_name, clf))
            
        # final_estimator = LogisticRegression(C=0.3, class_weight='balanced', random_state=seed)
        # final_estimator = LogisticRegression(C=0.2, class_weight='balanced', solver='liblinear', random_state=seed)
        final_estimator = LogisticRegression(
            penalty="elasticnet",
            solver="saga",      # 唯一支持 elasticnet 的求解器
            l1_ratio=0.5,       # 50% L1, 50% L2 (均衡策略)
            C=1.0,              # 正则力度 (C=0.3~0.5 都是最佳甜点区)
            class_weight="balanced",
            max_iter=5000,
            n_jobs=1,
            random_state=seed
        )
        print(f"   [Stacking] Meta Learner: LogisticRegression")

        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            stack_method='predict_proba',
            cv=5,
            n_jobs=1
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")

def coral_alignment(source, target, reg=1e-5):
    """
    对齐二阶统计量
    CORAL 域自适应，只在特征提取之后使用
    """
    n_s, d = source.shape
    # 计算源域和目标域的协方差矩阵
    cov_s = np.cov(source, rowvar=False) + np.eye(d) * reg
    cov_t = np.cov(target, rowvar=False) + np.eye(d) * reg
    # 消除源域原本的形状
    cov_s_sqrt = sqrtm(cov_s).real
    cov_s_inv_sqrt = inv(cov_s_sqrt)
    # 将特征分布扭曲成目标域的形状
    cov_t_sqrt = sqrtm(cov_t).real
    transform_matrix = np.dot(cov_s_inv_sqrt, cov_t_sqrt)
    return np.dot(source, transform_matrix)

def save_model(obj, name, model_dir):
    if not os.path.exists(model_dir): os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{name}.pkl")
    joblib.dump(obj, path)
    print(f"  [Saved] {path}")

def load_model(name, model_dir):
    path = os.path.join(model_dir, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    print(f"  [Loaded] {path}")
    return joblib.load(path)