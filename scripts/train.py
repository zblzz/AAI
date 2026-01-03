import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score


# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import get_dim_reducer, get_classifier, save_model
from utils.signed_log1p import SignedLog1pTransformer


def run_training(args):
    print(f"=== Training Start (Single Model + Global Alignment + OOF Threshold) ===")
    
    # ==========================================
    # 1. 加载数据
    # ==========================================
    train_file = os.path.join(args.data_dir, 'train.csv')
    test_cross_file = os.path.join(args.data_dir, 'test_cross_domain.csv')
    
    train_df = pd.read_csv(train_file)
    test_cross_df = pd.read_csv(test_cross_file)
    
    # 清洗列名
    train_df.columns = train_df.columns.str.strip()
    test_cross_df.columns = test_cross_df.columns.str.strip()
    
    target_col = 'label' if 'label' in train_df.columns else 'y'
    y_train = train_df[target_col].values
    X_train = train_df.drop(target_col, axis=1)
    
    # 提取 Cross 特征用于无监督对齐
    X_test_cross = test_cross_df[X_train.columns]
    
    print(f"Data Loaded. Train: {X_train.shape}, Test Cross: {X_test_cross.shape}")

    # ==========================================
    # 2. 全局无监督对齐 (Global Alignment)
    # ==========================================
    print("\n🚀 Step A: Global Alignment (Fit Scaler/PCA on Train + Test Cross)...")
    
    X_all = pd.concat([X_train, X_test_cross], axis=0)

    # 打印原始方差
    variances = X_all.var()
    print(f"   [Data Diagnosis] Original Min Var: {variances.min():.4f} | Max Var: {variances.max():.4f}")
    
    # 2.1 常量过滤 (threshold=0) - 必须先做这个，防止后续处理无效特征
    selector_var = VarianceThreshold(threshold=0.0) 
    selector_var.fit(X_all)
    kept_cols = X_train.columns[selector_var.get_support()]
    
    X_train_sub = pd.DataFrame(selector_var.transform(X_train), columns=kept_cols)
    X_test_cross_sub = pd.DataFrame(selector_var.transform(X_test_cross), columns=kept_cols)
    
    print(f"   -> Features reduced to {len(kept_cols)} (Constant features removed)")
    save_model(kept_cols.tolist(), 'robust_features', args.model_dir)

    # ============================================================
    # SignedLog1p 变换 (针对极大方差)
    # ============================================================
    print(f"   -> [Preprocessing] Applying SignedLog1p to compress dynamic range...")

    log_tf = SignedLog1pTransformer()
    # 注意：它通常不需要fit，但为了接口一致也可以fit
    X_train_sub = pd.DataFrame(log_tf.fit_transform(X_train_sub), columns=kept_cols)
    X_test_cross_sub = pd.DataFrame(log_tf.transform(X_test_cross_sub), columns=kept_cols)
    def check_finite(df, name):
        arr = df.to_numpy()
        n_nan = np.isnan(arr).sum()
        n_inf = np.isinf(arr).sum()
        if n_nan or n_inf:
            raise ValueError(f"{name} contains NaN/Inf (NaN={n_nan}, Inf={n_inf}).")
    check_finite(X_train_sub, "X_train_sub(after log)")
    check_finite(X_test_cross_sub, "X_test_cross_sub(after log)")
    save_model(log_tf, 'log_transformer', args.model_dir)
    
    # 为了确认效果，可以打印一下变换后的方差（可选）
    print(f"   -> Log-Transformed Max Var: {pd.concat([X_train_sub, X_test_cross_sub]).var().max():.4f}")
    # ============================================================

    # 重新合并用于后续的 Scaler 和 PCA 计算
    X_all_sub = pd.concat([X_train_sub, X_test_cross_sub], axis=0)

    # 2.2 全局标准化
    global_scaler = StandardScaler()
    global_scaler.fit(X_all_sub)
    save_model(global_scaler, 'scaler', args.model_dir) 
    
    X_train_scaled = global_scaler.transform(X_train_sub)
    X_all_scaled = global_scaler.transform(X_all_sub)
    
    # 2.3 全局降维 (Hybrid Fit)
    final_reducer = None
    X_train_ready = X_train_scaled 
    
    if args.dim_method != 'none':
        reducer_kwargs = vars(args).copy()
        for key in ['n_components', 'seed', 'data_dir', 'model_dir', 'clf_method', 'voters', 'voting_weights']:
             if key in reducer_kwargs: del reducer_kwargs[key]
             
        fresh_reducer = get_dim_reducer(args.dim_method, args.n_components, args.seed, **reducer_kwargs)
        
        if hasattr(fresh_reducer, 'steps'): # Pipeline
            selector_step = fresh_reducer.named_steps.get('selector') or fresh_reducer.named_steps.get('ensemble_selector')
            pca_step = fresh_reducer.named_steps.get('pca')
            
            print(f"   -> Hybrid Fitting: Selector on Train, PCA on All...")
            selector_step.fit(X_train_scaled, y_train) # 监督部分只看 Train
            
            X_all_selected = selector_step.transform(X_all_scaled)
            pca_step.fit(X_all_selected) # 无监督部分看 All
            
            final_reducer = Pipeline([('selector', selector_step), ('pca', pca_step)])
            X_train_ready = final_reducer.transform(X_train_scaled)
        else: # Simple PCA
            print(f"   -> Simple PCA: Fitting on All...")
            fresh_reducer.fit(X_all_scaled)
            final_reducer = fresh_reducer
            X_train_ready = final_reducer.transform(X_train_scaled)
            
        save_model(final_reducer, 'dim_reducer', args.model_dir)
    else:
        save_model(None, 'dim_reducer', args.model_dir)

    # ==========================================
    # 3. 准备分类器
    # ==========================================
    # if args.clf_method == 'stacking':
    #     voters_list = [v.strip() for v in args.voters.split(',')]
    #     estimators = [(v, get_classifier(v, args.seed)) for v in voters_list]
    #     meta_learner = LogisticRegression(
    #         penalty="elasticnet", 
    #         solver="saga", 
    #         l1_ratio=0.5, 
    #         C=0.5, 
    #         class_weight="balanced", 
    #         max_iter=5000, n_jobs=-1, random_state=args.seed)
    #     clf = StackingClassifier(estimators=estimators, final_estimator=meta_learner, stack_method='predict_proba', cv=5, n_jobs=-1)
    # else:
    clf = get_classifier(args.clf_method, args.seed, voters=args.voters)

    # ==========================================
    # 4. CV 质检 (Sanity Check)
    # ==========================================
    # print("\n🔍 Step D: Running CV Sanity Check (Transductive Setting)...")
    # cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=args.seed)
    # scoring = {'acc': 'accuracy', 'auc': 'roc_auc'}
    # cv_results = cross_validate(clf, X_train_ready, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    # mean_acc = cv_results['test_acc'].mean()
    # mean_auc = cv_results['test_auc'].mean()

    # print(f"CV Accuracy : {mean_acc:.4f} (+/- {cv_results['test_acc'].std()*2:.4f})")
    # print(f"CV AUC Score: {mean_auc:.4f} (+/- {cv_results['test_auc'].std()*2:.4f})")
    print("\n🔍 Step D: Running CV (NO-LEAK, train-only pipeline)...")

    # 只用训练集做 CV，所有 fit 都在 fold 内完成
    reducer_kwargs = vars(args).copy()
    for key in ['n_components', 'seed', 'data_dir', 'model_dir', 'clf_method', 'voters', 'voting_weights']:
        reducer_kwargs.pop(key, None)

    dim_reducer_for_cv = get_dim_reducer(args.dim_method, args.n_components, args.seed, **reducer_kwargs)

    cv_pipeline = Pipeline([
        ('var0', VarianceThreshold(threshold=0.0)),
        ('log', SignedLog1pTransformer()),
        ('scaler', StandardScaler()),
        ('reducer', dim_reducer_for_cv),
        ('clf', get_classifier(args.clf_method, args.seed, voters=args.voters)),
    ])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=args.seed)
    scoring = {'acc': 'accuracy', 'auc': 'roc_auc'}
    cv_results = cross_validate(cv_pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    mean_acc = cv_results['test_acc'].mean()
    mean_auc = cv_results['test_auc'].mean()
    print(f"CV Accuracy (no-leak): {mean_acc:.4f} (+/- {cv_results['test_acc'].std()*2:.4f})")
    print(f"CV AUC (no-leak)     : {mean_auc:.4f} (+/- {cv_results['test_auc'].std()*2:.4f})")

    # ==========================================
    # 5. OOF 阈值计算 & 全量训练
    # ==========================================
    # print("\n🚀 Step E: Maximizing Metric on OOF Probabilities...")

    # print("   -> Calculating OOF probabilities (5-Fold Stratified) [NO-LEAK end-to-end]...")
    # cv_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # # ✅ 用端到端 pipeline 计算 OOF 概率（包含 var/log/scaler/reducer/clf 的 fold 内拟合）
    # oof_probs = cross_val_predict(
    #     cv_pipeline,
    #     X_train,
    #     y_train,
    #     cv=cv_oof,
    #     method='predict_proba',
    #     n_jobs=-1
    # )[:, 1]

    # thresholds = np.linspace(0.1, 0.9, 801)
    # best_threshold, best_score = 0.5, -1.0
    # metric_func = balanced_accuracy_score
    # metric_name = "Balanced Acc"

    # for th in thresholds:
    #     preds = (oof_probs >= th).astype(int)
    #     score = metric_func(y_train, preds)
    #     if score > best_score:
    #         best_score, best_threshold = score, th

    # print(f"   Optimization Metric: {metric_name}")
    # print(f"   Best Threshold Found: {best_threshold:.4f} (Score: {best_score:.4f})")
    # print(f"   (Reference: Train Pos Rate was {np.mean(y_train):.4f})")

    # save_model(float(best_threshold), 'threshold', args.model_dir)

    # ==========================================
    # 5. 阈值计算 & 全量训练
    # ==========================================
    print("\n🚀 Step E: Threshold Calibration & Final Training...")

    print("   -> Fitting final model on full data (global-aligned features)...")
    clf.fit(X_train_ready, y_train)
    save_model(clf, 'model_unified', args.model_dir)

    # Pipeline 不能包含 None step
    steps = [
        ('var0', selector_var),
        ('log', log_tf),
        ('scaler', global_scaler),
    ]
    if final_reducer is not None:
        steps.append(('reducer', final_reducer))
    steps.append(('clf', clf))

    final_pipeline = Pipeline(steps)
    save_model(final_pipeline, 'model', args.model_dir)
    print("  [Saved] model.pkl")

    # pos_rate 阈值（与你现在一致）
    train_probs_final = final_pipeline.predict_proba(X_train)[:, 1]
    pos_rate = float(np.mean(y_train))
    percentile = 100.0 * (1.0 - pos_rate)
    best_threshold = float(np.percentile(train_probs_final, percentile))
    save_model(best_threshold, 'threshold', args.model_dir)

    # ✅ 自检：训练集上预测正例率应接近 pos_rate
    preds_train = (train_probs_final >= best_threshold).astype(int)
    print(f"   Train Pos Rate                   : {pos_rate:.4f}")
    print(f"   Threshold (pos_rate percentile)  : {best_threshold:.4f} (at {percentile:.1f}th)")
    print(f"   [Self-Check] Train Pred Pos Rate : {preds_train.mean():.4f}")

    if isinstance(clf, StackingClassifier):
        meta_model = clf.final_estimator_
        if hasattr(meta_model, 'coef_'):
            print(f"   Meta Weights: {meta_model.coef_[0]}")

    print("\n=== Training Done ===")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(base_dir / 'data'))
    parser.add_argument('--model_dir', default=str(base_dir / 'models'))

    ## default 里保存 SOTA
    parser.add_argument('--dim_method', default='ensemble_spca')
    parser.add_argument('--clf_method', default='stacking')
    parser.add_argument('--voters', type=str, default="lr,svm_calib,xgb")
    parser.add_argument('--n_components', type=int, default=120)
    parser.add_argument('--spca_k', type=int, default=3000)
    # parser.add_argument('--drop_adv_n', type=int, default=0) # 默认不丢弃
    # parser.add_argument('--var_threshold', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ens_verbose', action='store_true',
                        help="Print verbose logs inside EnsembleSelector (may be noisy under parallel CV).")
    
    parser.add_argument('--voting_weights', type=str, default=None)
    parser.add_argument('--ens_l1_c', type=float, default=1.0)
    parser.add_argument('--ens_mi_k', type=int, default=3)
        
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    run_training(args)


