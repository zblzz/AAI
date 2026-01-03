import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 路径修复，确保能导入 utils
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from utils.utils import load_model

def get_predictions(pipeline, threshold, data_path, task_name="Task"):
    if not os.path.exists(data_path):
        print(f"⚠️ File not found: {data_path}")
        return None

    print(f"\n🚀 Processing {task_name}: {os.path.basename(data_path)}")
    df = pd.read_csv(data_path)
    # 1. 清洗列名
    df.columns = df.columns.str.strip()

    # 2. 剔除无关列
    drop_cols = [c for c in ['label', 'y', 'id', 'ID'] if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    # 3. [关键安全检查] 确保特征数量正确
    # Pipeline 的第一步 (var0) 期望的特征数
    expected_features = pipeline.named_steps['var0'].n_features_in_
    if df.shape[1] != expected_features:
        print(f"⚠️ Warning: Feature dimension mismatch!")
        print(f"   Expected: {expected_features}, Got: {df.shape[1]}")
        # 这里如果维度不对，通常会直接报错，打印出来方便调试
    
    # 4. 预测
    try:
        probs = pipeline.predict_proba(df)[:, 1]
        preds = (probs >= float(threshold)).astype(int)

        print(f"📊 {task_name} Stats:")
        print(f"   Mean Prob: {probs.mean():.4f} | Std: {probs.std():.4f}")
        print(f"   Threshold: {float(threshold):.4f}")
        print(f"   Predicted: Pos={int(preds.sum())} ({preds.mean()*100:.1f}%), Neg={len(preds)-int(preds.sum())}")
        return preds
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return None

def generate_submission_file(preds, output_path):
    if preds is None:
        return
    pd.DataFrame(preds, columns=['y-pred']).to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path} (Rows: {len(preds)})")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=str(base_dir / 'data'))
    parser.add_argument('--model_dir', default=str(base_dir / 'models'))
    parser.add_argument('--output_dir', default=str(base_dir / 'output'))
    parser.add_argument('--team_id', default="5")
    parser.add_argument('--leader_name', default="Zoubolin")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Inference Start (Adaptive Threshold) ===")

    pipeline = load_model('model', args.model_dir)
    threshold = load_model('threshold', args.model_dir)
    # threshold=0.5
    print(f"✅ Loaded model.pkl")
    print(f"✅ Loaded threshold: {float(threshold):.4f}")

    task1_input = os.path.join(args.data_dir, 'test_in_domain.csv')
    task2_input = os.path.join(args.data_dir, 'test_cross_domain.csv')

    out1 = os.path.join(args.output_dir, f"{args.team_id}_{args.leader_name}_pred_in_domain.csv")
    out2 = os.path.join(args.output_dir, f"{args.team_id}_{args.leader_name}_pred_cross_domain.csv")

    preds1 = get_predictions(pipeline, threshold, task1_input, "Task 1 (In-Domain)")
    generate_submission_file(preds1, out1)

    preds2 = get_predictions(pipeline, threshold, task2_input, "Task 2 (Cross-Domain)")
    generate_submission_file(preds2, out2)

    print("\n=== Inference Done ===")