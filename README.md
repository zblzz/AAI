# AAI Project


本项目提供训练与推理脚本，用于二分类任务：
- **Task 1**：in-domain 测试集
- **Task 2**：cross-domain 测试集（目标域分布不同）

实现包含：
- **常量特征过滤**（VarianceThreshold=0）
- **SignedLog1p** 压缩长尾/极大方差特征
- **全局对齐（transductive）**：在 `train + test_cross` 上拟合 scaler 与 PCA（无标签）
- **维度约简**：`ensemble_spca`（监督 selector + 无监督 PCA 的 hybrid fit）
- **分类器**：stacking（默认 voters: `rf,svm_calib,xgb`）
- **阈值校准**：在最终 `final_pipeline` 的训练集预测概率上按训练先验（pos rate）取分位点阈值，输出 `threshold.pkl`
- **提交文件**：输出 0/1 标签，列名必须为 `y-pred`

> 注意：本项目默认使用 **transductive alignment**（无监督使用 `test_cross_domain.csv` 的特征分布）。如果课程/作业规则禁止在训练阶段使用测试集特征分布，请自行修改 `scripts/train.py` 的 Step A（只用 train 拟合 scaler/PCA）。

---

## 目录结构

```text
AAI/project/
  data/
    train.csv
    test_in_domain.csv
    test_cross_domain.csv
  models/
    *.pkl
  output/
    *_pred_in_domain.csv
    *_pred_cross_domain.csv
  scripts/
    train.py
    infer.py
    viz_data.py
    viz_pca_2d.py
  utils/
    utils.py
    ensemble_selector.py
    signed_log1p.py
  requirements.txt
  README.md
```

---

## 环境准备（Conda）

### 1) 创建环境
```bash
conda create -n AAI python=3.9 -y
conda activate AAI
```

### 2) 安装依赖
```bash
pip install -r requirements.txt
```

### 3) 验证
```bash
python -c "import numpy, pandas, sklearn, xgboost; print('OK')"
```

---

## 快速开始

### 训练（生成模型与阈值）
在项目根目录执行：

```bash
python scripts/train.py \
  --dim_method ensemble_spca \
  --clf_method stacking \
  --voters rf,svm_calib,xgb \
  --spca_k 2500 \
  --n_components 100
```

训练完成后会在 `models/` 下生成（示例）：
- `robust_features.pkl`：保留特征列表（常量列过滤后）
- `log_transformer.pkl`：SignedLog1p 变换器
- `scaler.pkl`：StandardScaler（按全局对齐拟合）
- `dim_reducer.pkl`：降维器（selector + PCA）
- `model_unified.pkl`：分类器
- `final_pipeline.pkl`：端到端推理流水线（推荐推理只用它）
- `threshold.pkl`：阈值（用于输出 0/1 标签）

### 推理（生成提交文件 0/1）
```bash
python scripts/infer.py
```

默认输出到：
- `output/5_Zoubolin_pred_in_domain.csv`
- `output/5_Zoubolin_pred_cross_domain.csv`

> 你可以在 `scripts/infer.py` 内修改默认 `team_id / leader_name`，或扩展为命令行参数。

---

## 脚本说明

### `scripts/train.py`
训练脚本，主要步骤：
1. 读取 `data/train.csv` 与 `data/test_cross_domain.csv`
2. 全局对齐：
   - 常量特征过滤（fit on train + test_cross）
   - SignedLog1p 压缩动态范围（train/test_cross 同变换）
   - StandardScaler（fit on train + test_cross）
3. Hybrid 降维：
   - selector 在 train 上监督拟合
   - PCA 在 train + test_cross 的 selector 输出上无监督拟合
4. no-leak CV sanity check（仅用于评估，不参与最终模型保存）
5. 最终训练与阈值校准：
   - 训练最终模型并保存 `final_pipeline.pkl`
   - 在训练集上预测概率，并按训练正例率分位点生成阈值 `threshold.pkl`


#### 主要参数
- `--data_dir`：数据目录（默认 `./data`）
- `--model_dir`：模型输出目录（默认 `./models`）
- `--dim_method`：降维方法  
  - `ensemble_spca`：推荐
  - `none`：不降维
- `--n_components`：PCA 输出维度（如 80/100/120）
- `--spca_k`：selector 预选特征数（如 1500/2000/2500）
- `--clf_method`：分类方法（如 `stacking`）
- `--voters`：stacking 基学习器列表（逗号分隔），例：`rf,svm_calib,xgb`
- `--seed`：随机种子（默认 42）
- `--ens_verbose`：打印 EnsembleSelector 内部日志（可能较吵，CV 并行时不建议开）
- `--ens_l1_c`：EnsembleSelector 内 L1 LogisticRegression 的 C
- `--ens_mi_k`：EnsembleSelector 内 mutual information 选择的 k（小样本建议小一些）

---

### `scripts/infer.py`
推理脚本（生成提交 CSV）。逻辑：
1. 加载 `models/final_pipeline.pkl` 与 `models/threshold.pkl`
2. 对 `test_in_domain.csv` 与 `test_cross_domain.csv` 预测概率
3. 以阈值输出 0/1 标签，保存为单列 `y-pred`

输出格式：
- 单列
- 列名：`y-pred`
- 不包含 index

---

## 可视化脚本（可选）
- `scripts/viz_data.py`：数据分布查看
- `scripts/viz_pca_2d.py`：PCA 2D 可视化

---

## 常见问题（FAQ）

### 1) CV 很高但推理很极端（几乎全 0 或全 1）？
通常原因：
- 训练/推理预处理不一致（忘记用 log_transformer、scaler、dim_reducer 顺序不一致）
- 阈值与最终模型概率分布不匹配

本项目推荐使用 `final_pipeline.pkl` 做端到端推理，避免组件错配。

### 2) 警告：`X does not have valid feature names`？
是 sklearn 的提示（feature name tracking）。不影响结果，但可通过让自定义 transformer 返回 DataFrame 来消除。

### 3) 是否存在“泄露”？
- no-leak CV：`scripts/train.py` 的 Step D 使用 train-only pipeline，不使用 test 数据（推荐用于调参对比）。
- transductive alignment：最终训练阶段会使用 `test_cross_domain.csv` 的**无标签特征分布**做 scaler/PCA 对齐。是否允许取决于课程规则。

---

## 复现实验建议（参数网格）
可尝试以下组合（先看 no-leak CV，再做最终训练提交）：
- `--spca_k`: 1500 / 2000 / 2500
- `--n_components`: 60 / 80 / 100 / 120
- voters: `rf,svm_calib,xgb`（推荐）或 `rf,lr,xgb`

---

## License
仅用于课程/作业用途（按课程要求使用）。
