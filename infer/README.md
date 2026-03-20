# Cell2Sentence vLLM Inference

This directory contains standalone inference scripts for Cell2Sentence models using vLLM.

## Files

- `c2s_predict_vllm.py`: Python script for vLLM-based batch inference
- `run_infer_vllm.sh`: Shell script to launch inference with default configurations
- `convert_test_prompts.py`: Script to convert test JSONL prompts to match training format
- `run_convert_prompts.sh`: Shell script to run prompt conversion

## Usage

### Prompt Format Conversion

**Important**: If your test JSONL file has prompts in a different format than training, you need to convert them first.

The training format uses `PerturbationPromptFormatterFromJSONL` which expects prompts in this format:
```
Given the following cell sentence of {num_genes} expressed genes representing a {cell_type} cell's basal state, predict the cell sentence after applying the perturbation: {treatment}.
Control cell sentence: {gene_list}.

Perturbed cell sentence:
```

To convert test prompts to match training format:

```bash
# Using shell script
export INPUT_FILE=/path/to/test.jsonl
export OUTPUT_FILE=/path/to/converted_test.jsonl
export NUM_GENES=500
bash run_convert_prompts.sh

# Or using Python directly
python3 convert_test_prompts.py \
    --input_file /path/to/test.jsonl \
    --output_file /path/to/converted_test.jsonl \
    --num_genes 500
```

### Quick Start

```bash
# Set input file and run inference
export INPUT_FILE=/path/to/conversations.jsonl
bash run_infer_vllm.sh
```

### Advanced Usage

You can customize the inference by modifying the configuration in `run_infer_vllm.sh` or by setting environment variables:

```bash
# Set custom parameters
export INPUT_FILE=/path/to/conversations.jsonl
export OUTPUT_DIR=/path/to/output
export DATASET_ID=my_dataset
export CUDA_VISIBLE_DEVICES=0,1

# Run inference
bash run_infer_vllm.sh
```

### Direct Python Script Usage

You can also run the Python script directly:

```bash
python3 c2s_predict_vllm.py \
    --input_file /path/to/conversations.jsonl \
    --output_dir /path/to/output \
    --model_path /path/to/model \
    --dataset_id my_dataset \
    --batch_size 512 \
    --tensor_parallel_size 2 \
    --max_new_tokens 256 \
    --temperature 0.2 \
    --top_p 0.9
```

## Configuration Parameters

### Required Parameters

- `--input_file`: Path to input JSONL file with conversation format
- `--output_dir`: Directory to save inference results
- `--model_path`: Path to Cell2Sentence model (local path or HuggingFace model ID)

### Optional Parameters

- `--dataset_id`: Dataset identifier (default: "unknown")
- `--batch_size`: Batch size for vLLM inference (default: 256)
- `--tensor_parallel_size`: Number of GPUs for tensor parallelism (default: 1)
- `--gpu_memory_utilization`: GPU memory utilization 0.0-1.0 (default: 0.9)
- `--max_new_tokens`: Maximum tokens to generate (default: 20)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top_p`: Top-p sampling (default: 1.0)
- `--top_k`: Top-k sampling (default: -1, disabled)

## Input Format

The input JSONL file should contain conversation data in the following format:

```json
{
  "id": "sample_001",
  "conversations": [
    {
      "from": "system",
      "value": "You are a specialized AI assistant..."
    },
    {
      "from": "human",
      "value": "What is the cell type for genes: GENE1, GENE2, ...?"
    },
    {
      "from": "gpt",
      "value": "T cell"
    }
  ],
  "group": "test"
}
```

## Output Format

The inference script generates a JSON file with predictions:

```json
{
  "model_name": "vandijklab/C2S-Scale-Gemma-2-2B",
  "dataset_id": "my_dataset",
  "index": 0,
  "task_type": "cell type",
  "task_variant": "singlecell_openended",
  "question": "What is the cell type for genes: ...?",
  "ground_truth": "T cell",
  "predicted_answer": "T cell",
  "full_response": "T cell",
  "group": "test"
}
```

## Environment Setup

The script requires:
- Python 3.8+
- vLLM library
- CUDA-capable GPUs
- Cell2Sentence model files

Make sure to activate the conda environment before running:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vLLM
```

## Notes

- The script automatically handles vLLM engine version compatibility
- Softcapping in model config is automatically disabled for vLLM compatibility
- The script supports both local model paths and HuggingFace model IDs
- GPU memory utilization can be adjusted based on available resources
- **For perturbation response prediction**: Ensure test prompts match the training format. Use `convert_test_prompts.py` if needed.
- The conversion script extracts information from `gene_lists.pre_perturbation_genes` and `metadata` fields
- The `num_genes` parameter should match the value used during training (default: 500)

