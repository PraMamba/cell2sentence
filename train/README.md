# Cell2Sentence 扰动响应预测训练

本目录包含用于训练 Cell2Sentence 模型进行扰动响应预测的脚本。

## 概述

该脚本改编自 `tutorials/c2s_tutorial_10_perturbation_response_prediction.ipynb`，专门适配了 JSONL 格式的扰动数据。

### 主要功能

- 从 JSONL 文件加载预配对的控制和扰动基因列表
- 使用自定义 `PerturbationPromptFormatterFromJSONL` 格式化训练数据
- 微调预训练的 Cell2Sentence 模型进行扰动响应预测
- 支持多种训练参数配置

## 数据格式要求

输入的 JSONL 文件每行应包含以下结构：

```json
{
  "id": "unique_sample_id",
  "data_type": "perturb_qa_topgenes",
  "conversations": [...],
  "metadata": {
    "perturbation_id": "...",
    "cell_type": "B cell",
    "treatment": "Edaravone",
    "perturbation_type": "drug",
    "description": "...",
    "perturbation_description": "...",
    "sample_index": 0,
    "pre_genes_count": 500,
    "post_genes_count": 500
  },
  "gene_lists": {
    "pre_perturbation_genes": ["GENE1", "GENE2", ...],
    "post_perturbation_genes": ["GENE1", "GENE2", ...]
  }
}
```

### 必需字段

- `gene_lists.pre_perturbation_genes`: 控制状态（扰动前）的基因列表
- `gene_lists.post_perturbation_genes`: 扰动状态（扰动后）的基因列表
- `metadata.treatment`: 扰动名称（如药物名称、基因名称等）
- `metadata.cell_type`: 细胞类型（可选，但推荐包含）

## 使用方法

### 基本用法

```bash
python c2s_perturbation_response_prediction.py \
    --input_file /path/to/your/perturbation_data.jsonl \
    --model_name_or_path vandijklab/C2S-Scale-Pythia-1b-pt \
    --output_dir ./output \
    --num_genes 500 \
    --epochs 3 \
    --batch_size 2
```

### 完整参数说明

#### 必需参数

- `--input_file`: 输入 JSONL 文件的路径

#### 可选参数

**模型和输出**
- `--model_name_or_path`: 预训练模型名称或路径（默认: `vandijklab/C2S-Scale-Pythia-1b-pt`）
- `--output_dir`: 输出目录（默认: `./output`）

**训练参数**
- `--num_genes`: 每个细胞句子使用的基因数量（默认: 500）
- `--epochs`: 训练轮数（默认: 3）
- `--batch_size`: 每个设备的批次大小（默认: 2）
- `--gradient_accumulation_steps`: 梯度累积步数（默认: 4）
- `--learning_rate`: 学习率（默认: 1e-5）
- `--max_steps`: 最大训练步数（如果设置，会覆盖 epochs）
- `--logging_steps`: 每 N 步记录一次日志（默认: 50）
- `--eval_steps`: 每 N 步评估一次（默认: 100）
- `--save_steps`: 每 N 步保存一次检查点（默认: 200）
- `--seed`: 随机种子（默认: 1234）
- `--val_split`: 验证集比例（默认: 0.1）

### 示例脚本

参见 `run_train.sh` 获取完整的训练示例。

```bash
bash run_train.sh
```

## 输出文件

训练完成后，输出目录将包含：

```
output/
├── csdata/                          # CSData 对象
│   └── perturbation_data/
├── model/                           # 初始模型副本
│   └── perturbation_model/
└── YYYY-MM-DD-HH_MM_SS_finetune_perturbation_prediction/
    ├── checkpoint-XXX/             # 训练检查点
    ├── data_split_indices_dict.pkl # 数据划分索引
    └── trainer_state.json          # 训练状态
```

## 推理使用

训练完成后，可以使用微调后的模型进行推理：

```python
import cell2sentence as cs
from transformers import AutoModelForCausalLM

# 加载微调后的模型
checkpoint_path = "./output/YYYY-MM-DD-HH_MM_SS_finetune_perturbation_prediction/checkpoint-XXX"

csmodel = cs.CSModel(
    model_name_or_path=checkpoint_path,
    save_dir="./inference",
    save_name="perturbation_predictor"
)

model = AutoModelForCausalLM.from_pretrained(
    csmodel.save_path,
    trust_remote_code=True
).to("cuda")

# 准备输入 prompt
prompt = """Given the following cell sentence of 500 expressed genes representing a B cell cell's basal state, predict the cell sentence after applying the perturbation: Edaravone.
Control cell sentence: MALAT1 B2M EEF1A1 RPL10 RPL41 ...

Perturbed cell sentence:"""

# 生成预测
prediction = csmodel.generate_from_prompt(
    model=model,
    prompt=prompt,
    max_num_tokens=2000
)

print("Predicted perturbed cell sentence:")
print(prediction)
```

## 与 Tutorial 的主要区别

1. **数据输入格式**：
   - Tutorial: 从 AnnData 对象创建，需要在训练时配对 control 和 perturbed cells
   - 本脚本: 从 JSONL 文件读取已配对的数据

2. **Prompt Formatter**：
   - Tutorial: `PerturbationPromptFormatter` 从全局 control pool 中随机选择 control cell
   - 本脚本: `PerturbationPromptFormatterFromJSONL` 直接使用预配对的数据

3. **基因列表处理**：
   - Tutorial: 从 AnnData 的表达矩阵中提取 top-k 基因
   - 本脚本: 直接从 JSONL 的基因列表中选取前 k 个基因

## 推荐的训练配置

根据数据集大小和硬件资源，推荐以下配置：

### 小规模数据集 (< 10K 样本)
```bash
--num_genes 500 \
--epochs 5 \
--batch_size 4 \
--gradient_accumulation_steps 2 \
--learning_rate 1e-5
```

### 中等规模数据集 (10K - 100K 样本)
```bash
--num_genes 500 \
--epochs 3 \
--batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5
```

### 大规模数据集 (> 100K 样本)
```bash
--num_genes 500 \
--epochs 2 \
--batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-6
```

## 注意事项

1. **GPU 内存**: 训练需要足够的 GPU 内存。如果遇到 OOM 错误，可以：
   - 减小 `--batch_size`
   - 增大 `--gradient_accumulation_steps`
   - 减小 `--num_genes`

2. **训练时间**: 训练时间取决于数据集大小、模型大小和硬件。一般来说：
   - 1B 参数模型在 A100 GPU 上训练 10K 样本约需 1-2 小时

3. **数据质量**: 确保输入数据的质量：
   - 基因名称应标准化（如使用 HUGO 符号）
   - 控制和扰动样本应合理配对
   - 基因列表应按表达量排序（从高到低）

## 故障排查

### 问题: CUDA out of memory
**解决方案**: 减小批次大小或使用梯度检查点
```bash
--batch_size 1 --gradient_accumulation_steps 8
```

### 问题: Training loss 不下降
**解决方案**: 调整学习率或检查数据质量
```bash
--learning_rate 5e-6
```

### 问题: 验证集表现差
**解决方案**: 增加训练轮数或数据增强
```bash
--epochs 5
```

## 参考

- 原始 Tutorial: `tutorials/c2s_tutorial_10_perturbation_response_prediction.ipynb`
- Cell2Sentence 文档: https://github.com/vandijklab/cell2sentence
- Transformers 文档: https://huggingface.co/docs/transformers

