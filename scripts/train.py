import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import get_dim_reducer, get_classifier, save_model

def run_training(args):
    print(f"=== Training Start (Method: {args.dim_method} + {args.clf_method}) ===")
    
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
    
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    print(f"Data Loaded. Shape: {X_train.shape}")
    print(f"Class Balance: Pos={num_pos}, Neg={num_neg}")

    # ==========================================
    # 2. 特征工程 Step A: 方差过滤
    # ==========================================
    print(f"Step A: Variance Thresholding (threshold={args.var_threshold})...")
    selector = VarianceThreshold(threshold=args.var_threshold)
    X_train_var = selector.fit_transform(X_train) 
    kept_cols = X_train.columns[selector.get_support()]
    
    X_train_sub = pd.DataFrame(X_train_var, columns=kept_cols)
    X_test_cross_sub = test_cross_df[kept_cols]
    
    print(f"   -> Features reduced from {X_train.shape[1]} to {X_train_sub.shape[1]}")

    # ==========================================
    # 3. 特征工程 Step 2: 标准化
    # ==========================================
    print("Step 2: Standardization...")
    temp_scaler = StandardScaler()
    # 仅用于 Step B 计算，后续 Pipeline 会重做
    X_train_adv_scaled = pd.DataFrame(temp_scaler.fit_transform(X_train_sub), columns=kept_cols)
    X_test_adv_scaled = pd.DataFrame(temp_scaler.transform(X_test_cross_sub), columns=kept_cols)

    # ==========================================
    # 4. 特征工程 Step B: 对抗性筛选 (默认为 0，保留全量)
    # ==========================================
    if args.drop_adv_n > 0:
        print(f"Step B: Adversarial Feature Selection (Dropping top {args.drop_adv_n})...")
        adv_X = pd.concat([X_train_adv_scaled, X_test_adv_scaled], axis=0)
        adv_y = np.array([0]*len(X_train_adv_scaled) + [1]*len(X_test_adv_scaled))
        
        adv_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=args.seed, n_jobs=-1)
        adv_clf.fit(adv_X, adv_y)
        
        imp = pd.Series(adv_clf.feature_importances_, index=kept_cols)
        drop_cols = imp.nlargest(args.drop_adv_n).index
        robust_features = [c for c in kept_cols if c not in drop_cols]
        print(f"   -> Dropped {args.drop_adv_n} features.")
    else:
        print(f"Step B: Skipping Adversarial Drop (drop_adv_n=0). Keeping all features.")
        robust_features = kept_cols.tolist()
    
    print(f"   -> Final Feature Count: {len(robust_features)}")
    save_model(robust_features, 'robust_features', args.model_dir)
    X_train_final = X_train_sub[robust_features]

    # ==========================================
    # 5. 构建模型 (Stacking with ElasticNet)
    # ==========================================
    # 准备降维器参数，剔除冲突参数
    reducer_kwargs = vars(args).copy()
    for key in ['n_components', 'seed', 'data_dir', 'model_dir', 'clf_method', 'voters', 'voting_weights']:
        if key in reducer_kwargs:
            del reducer_kwargs[key]

    reducer = get_dim_reducer(args.dim_method, args.n_components, args.seed, **reducer_kwargs)
    clf = get_classifier(args.clf_method, args.seed, voters=args.voters)

    print("\n" + "-"*30)
    print(f"Running Repeated CV with {args.dim_method} + {args.clf_method}...")
    print("-" * 30)
        
    pipeline_steps = [('scaler', StandardScaler())]
    if reducer:
        pipeline_steps.append(('reducer', reducer))
    pipeline_steps.append(('clf', clf))
    
    model_pipeline = Pipeline(pipeline_steps)

    # 5x5 Repeated CV
    scoring = {'acc': 'accuracy', 'auc': 'roc_auc'}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=args.seed)

    cv_results = cross_validate(model_pipeline, X_train_final, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    mean_acc = cv_results['test_acc'].mean()
    mean_auc = cv_results['test_auc'].mean()

    print(f"CV Accuracy : {mean_acc:.4f} (+/- {cv_results['test_acc'].std()*2:.4f})")
    print(f"CV AUC Score: {mean_auc:.4f} (+/- {cv_results['test_auc'].std()*2:.4f})")
    print("-" * 30 + "\n")

    # ==========================================
    # 6. 全量训练
    # ==========================================
    print(f"Retraining on FULL dataset for export...")
    
    final_scaler = StandardScaler()
    X_train_scaled_final = final_scaler.fit_transform(X_train_final)
    save_model(final_scaler, 'scaler', args.model_dir)
    
    if reducer:
        print(f"   Fitting {args.dim_method} on full data...")
        X_train_reduced = reducer.fit_transform(X_train_scaled_final, y_train)
        save_model(reducer, 'dim_reducer', args.model_dir)
    else:
        X_train_reduced = X_train_scaled_final
        save_model(None, 'dim_reducer', args.model_dir)
        
    print(f"   Fitting classifier on full data...")
    clf.fit(X_train_reduced, y_train)
    save_model(clf, 'model_unified', args.model_dir)
    
    # 打印权重分析
    if isinstance(clf, StackingClassifier):
        print_stacking_weights(clf)

    # ===============================================================
    # 7. 伪标签 (Pseudo-Labeling)
    # ===============================================================
    print("\n" + "="*40)
    print("🚀 FORCE START: Pseudo-Labeling Strategy")
    print("="*40)

    try:
        X_test_full_ready = final_scaler.transform(X_test_cross_sub[robust_features])
    except Exception as e:
        print(f"   [Error] Preprocessing test data failed: {e}")
        X_test_full_ready = None

    if X_test_full_ready is not None:
        if reducer:
            X_test_reduced = reducer.transform(X_test_full_ready)
        else:
            X_test_reduced = X_test_full_ready

        probs_test = clf.predict_proba(X_test_reduced)[:, 1]
        
        # 自适应阈值选择：从严到宽，确保能选出样本
        selected_indices = []
        for threshold in [0.95, 0.90, 0.85]:
            high_conf_idx = np.where((probs_test >= threshold) | (probs_test <= (1 - threshold)))[0]
            if len(high_conf_idx) >= 20: 
                selected_indices = high_conf_idx
                print(f"   [Pseudo] Threshold selected: {threshold}/{1-threshold:.2f}")
                break
        
        # 兜底策略：如果还是太少，放宽到0.85强行选
        if len(selected_indices) == 0 and len(high_conf_idx) > 0:
            selected_indices = high_conf_idx
            print(f"   [Pseudo] Threshold relaxed to: 0.85 (Found few samples)")

        print(f"   [Pseudo] High Confidence Samples Found: {len(selected_indices)}")
        
        if len(selected_indices) > 0:
            pseudo_X = X_test_reduced[selected_indices]
            # 生成伪标签：>0.5设为1，<=0.5设为0
            pseudo_y = (probs_test[selected_indices] > 0.5).astype(int)
            
            # 合并数据
            X_aug = np.vstack([X_train_reduced, pseudo_X])
            y_aug = np.concatenate([y_train, pseudo_y])
            
            print(f"   [Pseudo] Retraining with Augmented Data: {len(y_train)} -> {len(y_aug)} samples")
            
            # 重新训练模型（覆盖原模型）
            clf.fit(X_aug, y_aug)
            save_model(clf, 'model_unified_pseudo', args.model_dir)
            print("   ✅ [Pseudo] Boosted Model Saved as 'model_unified_pseudo.pkl'!")
        else:
            print("   ⚠️ [Pseudo] No confident samples found. Keeping original model.")
            save_model(clf, 'model_unified_pseudo', args.model_dir)

    print("=== Training Done ===")

def print_stacking_weights(clf):
    print("\n" + "="*40)
    print("🕵️ Stacking Meta-Learner Weights Analysis")
    print("=" * 40)
    meta_model = clf.final_estimator_
    
    if hasattr(meta_model, 'coef_'):
        coefs = meta_model.coef_[0]
        base_names = list(clf.named_estimators_.keys())
        
        print(f"Meta Learner Intercept: {meta_model.intercept_[0]:.4f}")
        print("-" * 40)
        
        if len(coefs) == len(base_names) * 2: # 二分类且 output=predict_proba
            print(f"{'Base Learner':<15} | {'Class 0 Weight':<15} | {'Class 1 Weight':<15}")
            print("-" * 40)
            for i, name in enumerate(base_names):
                w0 = coefs[2*i]
                w1 = coefs[2*i+1]
                print(f"{name:<15} | {w0: .4f}          | {w1: .4f}")
        else:
            print(f"Raw Coefficients: {coefs}")
            print(f"Base Estimators: {base_names}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(base_dir / 'data'))
    parser.add_argument('--model_dir', default=str(base_dir / 'models'))

    ## default 里保存 SOTA
    parser.add_argument('--dim_method', default='ensemble_spca')
    parser.add_argument('--clf_method', default='stacking')
    parser.add_argument('--voters', type=str, default="lr,svm_calib,xgb")
    parser.add_argument('--n_components', type=int, default=80)
    parser.add_argument('--drop_adv_n', type=int, default=0) # 默认不丢弃
    parser.add_argument('--spca_k', type=int, default=2000)
    parser.add_argument('--var_threshold', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ens_verbose', action='store_true',
                        help="Print verbose logs inside EnsembleSelector (may be noisy under parallel CV).")

    # 兼容参数
    parser.add_argument('--voting_weights', type=str, default=None)
    parser.add_argument('--ens_l1_c', type=float, default=1.0)
    parser.add_argument('--ens_mi_k', type=int, default=3)
        
    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    run_training(args)