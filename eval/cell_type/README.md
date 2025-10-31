# Cell2Sentence 评估系统

## 📁 目录结构

```
Cell2Sentence/
└── A013/
    └── eval_results/
        └── singlecell_openended/          # 所有结果（预测和标准化）
            ├── singlecell_openended_predictions_*.json       # 标准化预测
            └── singlecell_openended_unmapped_celltypes_*.json # 未映射类型
└── D099/
    └── (same structure)
```

## 🔄 Pipeline 工作流

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. 原始数据 (.h5ad)                          │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              [c2s_predict.py] 模型预测                          │
│  - 加载 Cell2Sentence 模型                                       │
│  - 提取 top 200 genes 作为输入                                   │
│  - 生成细胞类型预测                                              │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              [c2s_predict.py] 模型预测                          │
│  - 生成预测并保存到 eval_results/singlecell_openended/         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│       [singlecell_openended_eval.py] 标准化                     │
│  - 读取预测文件                                                  │
│  - 应用 metadata_standard_mapping.py 映射                       │
│  - 覆盖保存标准化预测                                            │
│  - 记录未映射类型                                                │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│    eval_results/singlecell_openended/ (最终输出)               │
│  ├── predictions (standardized)                                 │
│  └── unmapped_celltypes                                         │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              [待实现] LLM as a Judge 语义评估                    │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 批量运行（推荐）

```bash
cd ~/cell2sentence/eval/cell_type
./run_eval.sh
```

### 手动分步运行

**步骤 1：生成预测**
```bash
python c2s_predict.py \
    --data_path /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/Processed_Data/A013_processed_sampled_w_cell2sentence.h5ad \
    --model_path vandijklab/C2S-Pythia-410m-cell-type-prediction \
    --output_dir /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/A013/eval_results/singlecell_openended \
    --dataset_id A013 \
    --n_genes 200 \
    --seed 1234
```

**步骤 2：标准化**
```bash
python singlecell_openended_eval.py \
    --predictions_file /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/A013/eval_results/singlecell_openended/singlecell_openended_predictions_*.json \
    --output_dir /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/A013/eval_results/singlecell_openended
```

## 📄 输出文件格式

### eval_results/singlecell_openended/ - 最终结果

#### 1. 标准化预测 (*_predictions_*.json)
```json
{
  "model_name": "vandijklab/C2S-Pythia-410m-cell-type-prediction",
  "dataset_id": "A013",
  "index": 0,
  "task_type": "cell type",
  "task_variant": "singlecell_openended",
  "question": "CD74 MALAT1 RPLP1 RPS8 ... (200 genes)",
  "ground_truth": "Naive B cell",           # 标准化后
  "predicted_answer": "Naive B cell",       # 标准化后
  "full_response": "naive B cell",
  "group": ""
}
```

注意：`ground_truth` 和 `predicted_answer` 已通过映射标准化

#### 2. 未映射类型 (*_unmapped_celltypes_*.json)
```json
{
  "CD4 T": {
    "count": 1,
    "indices": [123]
  }
}
```

## 🛠️ 核心脚本说明

### c2s_predict.py
**功能：** 使用 Cell2Sentence 模型生成细胞类型预测

**参数：**
- `--data_path` - .h5ad 文件路径
- `--model_path` - C2S 模型路径（支持 HuggingFace ID 或本地路径）
- `--output_dir` - 输出目录（eval_results/singlecell_openended/）
- `--dataset_id` - 数据集标识符（如 A013, D099）
- `--n_genes` - 用于预测的 top genes 数量（默认 200）
- `--seed` - 随机种子（默认 1234）

**输出：** `eval_results/singlecell_openended/singlecell_openended_predictions_TIMESTAMP.json`

### singlecell_openended_eval.py
**功能：** 对预测结果进行标准化，并记录未映射的细胞类型

**参数：**
- `--predictions_file` - 预测结果 JSON 文件路径
- `--output_dir` - 输出目录（eval_results/singlecell_openended/）

**输出：**
- `singlecell_openended_predictions_TIMESTAMP.json` - 标准化预测
- `singlecell_openended_unmapped_celltypes_TIMESTAMP.json` - 未映射类型

**注意：** 不生成评估指标，评估将由后续的 LLM as a judge 完成

### run_eval.sh
**功能：** 批量运行完整 pipeline

**配置：**
- `DATASETS` 数组 - 定义要处理的数据集
- `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR` - 目录配置
- `MODEL_PATH` - 模型路径
- `N_GENES`, `SEED` - 预测参数

## 📊 评估流程

### 当前：标准化处理
- 使用 `metadata_standard_mapping.py` 进行细胞类型名称标准化
- 记录未映射的细胞类型（用于后续补充映射字典）

### 计划：LLM as a Judge 评估
- 输入：`eval_results/singlecell_openended/*_predictions_*.json`
- 评估：ground_truth 和 predicted_answer 的语义一致性
- 输出：语义匹配评分和分析
- **注意：** 不再进行 exact match 评估，所有评估由 LLM as a judge 完成

## 🗂️ 文件清单

### 核心脚本
- ✅ `c2s_predict.py` - 预测脚本
- ✅ `singlecell_openended_eval.py` - 标准化和评估脚本
- ✅ `run_eval.sh` - Pipeline 编排脚本

### 工具模块
- ✅ `celltype_standardizer.py` - 细胞类型映射工具
- ✅ `convert_notebook_predictions.py` - Notebook 输出转换工具

### 文档
- 📄 `README.md` - 本文件（系统说明）
- 📄 `PIPELINE_CHANGES.md` - Pipeline 变更说明
- 📄 `UPDATE_LOG.md` - 更新日志
- 📄 `SUMMARY.txt` - 系统总结

## ⚙️ 配置

### 环境要求
```bash
conda activate Axolotl
```

### 关键路径
```bash
# 数据目录
/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/Processed_Data/

# 输出目录
/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/

# 模型路径
vandijklab/C2S-Pythia-410m-cell-type-prediction
# 或本地缓存：
# /data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Pythia-410m-cell-type-prediction/snapshots/...
```

### 映射字典
```python
# 来源：/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard/metadata_standard_mapping.py
CELL_TYPE_MAPPING = {
    # 1113 个映射规则
    "naive B cell": "Naive B cell",
    "CD4 T": "CD4+ T cell",
    ...
}
```

## 🔍 调试和验证

### 检查结果
```bash
# 查看标准化预测数量
jq '. | length' /path/to/eval_results/singlecell_openended/singlecell_openended_predictions_*.json

# 查看未映射类型
cat /path/to/eval_results/singlecell_openended/singlecell_openended_unmapped_celltypes_*.json | jq .
```

## 📝 注意事项

1. **目录用途**：
   - `eval_results/singlecell_openended/` - 所有结果（预测后直接标准化）

2. **数据流向**：
   - `c2s_predict.py` 直接将预测保存到 `eval_results/singlecell_openended/`
   - `singlecell_openended_eval.py` 读取该文件，标准化后覆盖保存
   - LLM as a judge 将读取 `eval_results/` 中的标准化结果

3. **文件命名**：
   - 所有输出文件包含时间戳，便于版本管理
   - 格式：`singlecell_openended_{type}_{timestamp}.json`

## 🎯 后续工作

- [ ] 实现 LLM as a judge 评估
- [ ] 添加可视化（混淆矩阵、性能图表）
- [ ] 处理更多数据集
- [ ] 优化未映射类型的处理流程

---
**维护：** AI Assistant  
**更新：** 2024-10-30  
**版本：** 2.0
