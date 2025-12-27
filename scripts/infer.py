import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入自定义工具库 (确保 load_model 能正常工作)
from utils.utils import load_model

def run_inference(args):
    print("=== Inference Start ===")
    
    # 1. 加载数据 (Task 1 & Task 2)
    test_in_file = os.path.join(args.data_dir, 'test_in_domain.csv')
    test_cross_file = os.path.join(args.data_dir, 'test_cross_domain.csv')
    
    if not os.path.exists(test_in_file) or not os.path.exists(test_cross_file):
        raise FileNotFoundError("Test files not found! Check your data directory.")
        
    df_in = pd.read_csv(test_in_file)
    df_cross = pd.read_csv(test_cross_file)
    
    # 清洗列名
    df_in.columns = df_in.columns.str.strip()
    df_cross.columns = df_cross.columns.str.strip()
    
    print(f"Loaded Test Data:")
    print(f"  In-Domain: {df_in.shape}")
    print(f"  Cross-Domain: {df_cross.shape}")

    # 2. 加载模型组件
    # 注意：我们要加载的是那个被伪标签增强过的 'model_unified_pseudo'
    # 如果没生成 pseudo 版，代码会自动回退去加载普通版
    
    # try:
    #     clf = load_model('model_unified_pseudo', args.model_dir)
    #     print("✅ Loaded model: model_unified_pseudo.pkl (The Boosted One!)")
    # except:
    #     print("⚠️ Pseudo model not found, loading standard model...")
    clf = load_model('model_unified', args.model_dir)

    scaler = load_model('scaler', args.model_dir)
    
    # 加载 robust_features 列表
    feature_path = os.path.join(args.model_dir, 'robust_features.pkl')
    if os.path.exists(feature_path):
        with open(feature_path, 'rb') as f:
            robust_features = pickle.load(f)
        print(f"✅ Loaded feature list: {len(robust_features)} features")
    else:
        print("❌ Feature list not found! Using all columns.")
        robust_features = df_in.columns.tolist()

    # 尝试加载降维器 (如果有)
    try:
        reducer = load_model('dim_reducer', args.model_dir)
        print("✅ Loaded Dim Reducer (SPCA)")
    except:
        reducer = None
        print("ℹ️ No Dim Reducer found (maybe None was saved).")

    # ==========================================
    # 3. 预测函数
    # ==========================================
    def predict_pipeline(df_input, task_name):
        print(f"\nProcessing {task_name}...")
        
        # A. 特征筛选 (Robust Features)
        X_sub = df_input[robust_features]
        
        # B. 标准化 (Scaler)
        # 注意：这里只能 transform，绝对不能 fit！
        X_scaled = scaler.transform(X_sub)
        
        # C. 降维 (Reducer)
        if reducer:
            X_reduced = reducer.transform(X_scaled)
        else:
            X_reduced = X_scaled
            
        # D. 预测 (Classifier)
        # 题目要求提交 0/1 标签
        y_pred = clf.predict(X_reduced)
        
        return y_pred

    # ==========================================
    # 4. 执行预测 & 保存
    # ==========================================
    
    # --- Task 1: In-Domain ---
    pred_in = predict_pipeline(df_in, "Task 1 (In-Domain)")
    
    # 保存 Task 1
    # 格式要求: pred_in_domain.csv, containing a single column y-pred
    out_in_path = os.path.join(args.output_dir, 'pred_in_domain.csv')
    pd.DataFrame({'y-pred': pred_in}).to_csv(out_in_path, index=False)
    print(f"Saved: {out_in_path}")

    # --- Task 2: Cross-Domain ---
    pred_cross = predict_pipeline(df_cross, "Task 2 (Cross-Domain)")
    
    # 保存 Task 2
    # 格式要求: pred_cross_domain.csv, containing a single column y-pred
    out_cross_path = os.path.join(args.output_dir, 'pred_cross_domain.csv')
    pd.DataFrame({'y-pred': pred_cross}).to_csv(out_cross_path, index=False)
    print(f"Saved: {out_cross_path}")

    print("\n=== Inference Done! Good Luck! ===")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(base_dir / 'data'))
    parser.add_argument('--model_dir', default=str(base_dir / 'models'))
    parser.add_argument('--output_dir', default=str(base_dir / 'output' ))
    
    args = parser.parse_args()
    run_inference(args)