#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perturbation Response Prediction Training Script

This script adapts the Cell2Sentence perturbation prediction task to work with
pre-formatted JSONL data containing paired control and perturbed gene lists.

Based on: tutorials/c2s_tutorial_10_perturbation_response_prediction.ipynb

Usage:
    python c2s_perturbation_response_prediction.py \
        --input_file /path/to/perturbation_data.jsonl \
        --model_name_or_path vandijklab/C2S-Scale-Pythia-1b-pt \
        --output_dir ./output \
        --num_genes 500 \
        --epochs 3 \
        --batch_size 2
"""

import os
import sys
import json
import random
import pickle
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

# Add parent directory to path for development mode
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from transformers import TrainingArguments, AutoModelForCausalLM
from tqdm import tqdm

# Cell2Sentence imports
import cell2sentence as cs
from cell2sentence.prompt_formatter import PromptFormatter


class PerturbationPromptFormatterFromJSONL(PromptFormatter):
    """
    Custom prompt formatter for perturbation prediction from pre-paired JSONL data.
    
    Unlike the tutorial version which pairs control and perturbed cells from a pool,
    this formatter works with pre-paired data where each record contains both
    control (pre-perturbation) and perturbed (post-perturbation) gene lists.
    """
    
    def __init__(
        self,
        task_name: str,
        input_prompt: str,
        answer_template: str,
        top_k_genes: int,
        jsonl_data: List[Dict]
    ):
        """
        Initialize the custom prompt formatter.
        
        Args:
            task_name: The name for this task
            input_prompt: The template for the model's input
            answer_template: The template for the model's expected response
            top_k_genes: The number of top genes to include in the cell sentence
            jsonl_data: List of dictionaries loaded from JSONL file
        """
        super().__init__()
        self.task_name = task_name
        self.input_prompt = input_prompt
        self.answer_template = answer_template
        self.top_k_genes = top_k_genes
        self.jsonl_data = jsonl_data
        assert top_k_genes > 0, "'top_k_genes' must be an integer > 0"
    
    def format_hf_ds(self, hf_ds):
        """
        Format the dataset by creating prompt-response pairs from JSONL data.
        
        This method extracts pre and post perturbation gene lists from the JSONL data
        and formats them according to the prompt templates.
        
        Args:
            hf_ds: Huggingface dataset (not used in this implementation, kept for API compatibility)
        
        Returns:
            Dataset: Formatted Huggingface Dataset with model_input and response columns
        """
        model_inputs_list = []
        responses_list = []
        
        print(f"Processing {len(self.jsonl_data)} perturbation samples...")
        
        for sample in tqdm(self.jsonl_data):
            # Extract gene lists from JSONL data
            gene_lists = sample.get('gene_lists', {})
            pre_genes = gene_lists.get('pre_perturbation_genes', [])
            post_genes = gene_lists.get('post_perturbation_genes', [])
            
            # Get perturbation metadata
            metadata = sample.get('metadata', {})
            treatment = metadata.get('treatment', 'Unknown')
            cell_type = metadata.get('cell_type', 'Unknown')
            perturbation_type = metadata.get('perturbation_type', 'drug')
            
            # Limit to top_k_genes
            control_genes = pre_genes[:self.top_k_genes]
            perturbed_genes = post_genes[:self.top_k_genes]
            
            # Format gene lists as space-separated strings
            control_sentence = ' '.join(control_genes)
            perturbed_sentence = ' '.join(perturbed_genes)
            
            # Get actual number of genes used
            num_genes = len(control_genes)
            
            # Format the model input string
            model_input_str = self.input_prompt.format(
                num_genes=num_genes,
                perturbation_name=treatment,
                cell_type=cell_type,
                control_cell_sentence=control_sentence
            )
            
            # Format the response string
            response_str = self.answer_template.format(
                perturbed_cell_sentence=perturbed_sentence
            )
            
            model_inputs_list.append(model_input_str)
            responses_list.append(response_str)
        
        # Create the final Hugging Face Dataset
        ds_split_dict = {
            "sample_type": [self.task_name] * len(model_inputs_list),
            "model_input": model_inputs_list,
            "response": responses_list,
        }
        ds = Dataset.from_dict(ds_split_dict)
        
        print(f"Created formatted dataset with {len(ds)} samples")
        return ds


def load_jsonl_data(input_file: str) -> List[Dict]:
    """
    Load perturbation data from JSONL file.
    
    Args:
        input_file: Path to JSONL file containing perturbation data
    
    Returns:
        List of dictionaries, each containing perturbation information
    """
    print(f"Loading data from: {input_file}")
    data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} perturbation samples")
    return data


def create_dummy_arrow_dataset(jsonl_data: List[Dict], vocabulary: Dict[str, int]) -> Dataset:
    """
    Create a dummy Arrow dataset compatible with CSData.
    
    Since we're using custom formatting via PerturbationPromptFormatterFromJSONL,
    we just need a minimal dataset that satisfies CSData requirements.
    
    Args:
        jsonl_data: List of dictionaries from JSONL
        vocabulary: Gene vocabulary dictionary
    
    Returns:
        Huggingface Dataset
    """
    # Create dummy cell sentences (these will be replaced by custom formatter)
    dummy_data = {
        'cell_sentence': [' '.join(sample['gene_lists']['pre_perturbation_genes'][:100]) 
                          for sample in jsonl_data],
        'organism': ['Homo sapiens'] * len(jsonl_data),
    }
    
    return Dataset.from_dict(dummy_data)


def create_gene_vocabulary(jsonl_data: List[Dict]) -> Dict[str, int]:
    """
    Create gene vocabulary from JSONL data.
    
    Args:
        jsonl_data: List of dictionaries from JSONL
    
    Returns:
        Dictionary mapping gene names to indices
    """
    all_genes = set()
    
    for sample in jsonl_data:
        gene_lists = sample.get('gene_lists', {})
        pre_genes = gene_lists.get('pre_perturbation_genes', [])
        post_genes = gene_lists.get('post_perturbation_genes', [])
        all_genes.update(pre_genes)
        all_genes.update(post_genes)
    
    # Create vocabulary with sorted genes
    vocabulary = {gene: idx for idx, gene in enumerate(sorted(all_genes))}
    
    print(f"Created vocabulary with {len(vocabulary)} unique genes")
    return vocabulary


def train_perturbation_model(
    input_file: str,
    model_name_or_path: str,
    output_dir: str,
    num_genes: int = 500,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    max_length: int = 8192,
    max_steps: Optional[int] = None,
    logging_steps: int = 50,
    logging_first_step: bool = True,
    logging_strategy: str = "steps",
    eval_steps: int = 100,
    eval_strategy: str = "steps",
    save_steps: int = 200,
    save_strategy: str = "steps",
    save_total_limit: int = 5,
    seed: int = 1234,
    val_split: float = 0.1,
    dataloader_num_workers: int = 0,
    bf16: bool = True,
    fp16: bool = False,
    gradient_checkpointing: bool = True,
    ddp_find_unused_parameters: bool = False,
    ddp_timeout: int = 30000,
    log_on_each_node: bool = False,
    do_train: bool = True,
    do_eval: bool = True,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    report_to: str = "none",
    overwrite_output_dir: bool = True,
    run_name: Optional[str] = None,
    deepspeed: Optional[str] = None,
    use_liger_kernel: bool = False,
):
    """
    Main training function for perturbation response prediction.
    
    Args:
        input_file: Path to JSONL file with perturbation data
        model_name_or_path: Pretrained model to start from
        output_dir: Directory to save outputs
        num_genes: Number of genes to use in cell sentences
        epochs: Number of training epochs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        lr_scheduler_type: Learning rate scheduler type
        max_length: Maximum sequence length
        max_steps: Maximum training steps (overrides epochs if set)
        logging_steps: Log every N steps
        logging_first_step: Log first step
        logging_strategy: Logging strategy
        eval_steps: Evaluate every N steps
        eval_strategy: Evaluation strategy
        save_steps: Save checkpoint every N steps
        save_strategy: Save strategy
        save_total_limit: Maximum number of checkpoints to keep
        seed: Random seed
        val_split: Validation split ratio
        dataloader_num_workers: Number of dataloader workers
        bf16: Use bfloat16
        fp16: Use float16
        gradient_checkpointing: Enable gradient checkpointing
        ddp_find_unused_parameters: DDP find unused parameters
        ddp_timeout: DDP timeout in seconds
        log_on_each_node: Log on each node
        do_train: Do training
        do_eval: Do evaluation
        load_best_model_at_end: Load best model at end
        metric_for_best_model: Metric for best model
        greater_is_better: Greater is better for metric
        report_to: Reporting destination
        overwrite_output_dir: Overwrite output directory
        run_name: Run name for logging
        deepspeed: DeepSpeed config file path
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSONL data
    jsonl_data = load_jsonl_data(input_file)
    
    # Split data into train and validation
    random.shuffle(jsonl_data)
    val_size = int(len(jsonl_data) * val_split)
    train_data = jsonl_data[val_size:]
    val_data = jsonl_data[:val_size]
    
    print(f"Split: {len(train_data)} train, {len(val_data)} validation samples")
    
    # Create gene vocabulary
    vocabulary = create_gene_vocabulary(jsonl_data)
    
    # Create dummy arrow dataset for CSData compatibility
    dummy_arrow_ds = create_dummy_arrow_dataset(jsonl_data, vocabulary)
    
    # Save CSData
    csdata_dir = os.path.join(output_dir, "csdata")
    os.makedirs(csdata_dir, exist_ok=True)
    
    # Save arrow dataset
    arrow_save_path = os.path.join(csdata_dir, "perturbation_data")
    dummy_arrow_ds.save_to_disk(arrow_save_path)
    
    # Create CSData object
    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=dummy_arrow_ds,
        vocabulary=vocabulary,
        save_dir=csdata_dir,
        save_name="perturbation_data",
        dataset_backend="arrow"
    )
    print(f"Created CSData: {csdata}")
    
    # Define prompt templates
    input_prompt_template = """Given the following cell sentence of {num_genes} expressed genes representing a {cell_type} cell's basal state, predict the cell sentence after applying the perturbation: {perturbation_name}.
Control cell sentence: {control_cell_sentence}.

Perturbed cell sentence:"""
    
    answer_template = "{perturbed_cell_sentence}."
    
    # Create custom prompt formatter
    task_name = "perturbation_prediction"
    prompt_formatter = PerturbationPromptFormatterFromJSONL(
        task_name=task_name,
        input_prompt=input_prompt_template,
        answer_template=answer_template,
        top_k_genes=num_genes,
        jsonl_data=train_data
    )
    
    # Format training data
    formatted_train_ds = prompt_formatter.format_hf_ds(None)
    
    # Format validation data
    val_prompt_formatter = PerturbationPromptFormatterFromJSONL(
        task_name=task_name,
        input_prompt=input_prompt_template,
        answer_template=answer_template,
        top_k_genes=num_genes,
        jsonl_data=val_data
    )
    formatted_val_ds = val_prompt_formatter.format_hf_ds(None)
    
    # Combine train and val for data_split_indices_dict
    # Use concatenate_datasets instead of + operator for Column objects
    formatted_all_ds = concatenate_datasets([formatted_train_ds, formatted_val_ds])
    
    # Create data split indices
    train_indices = list(range(len(formatted_train_ds)))
    val_indices = list(range(len(formatted_train_ds), len(formatted_all_ds)))
    
    data_split_indices_dict = {
        'train': train_indices,
        'val': val_indices,
        'test': val_indices[:min(100, len(val_indices))]  # Use subset of val as test
    }
    
    # Print sample formatted data
    print("\n=== Sample Formatted Data (for training) ===")
    print("Note: This is the Format-1 style prompt used for training.")
    print("The original JSONL contains conversations format (Format-2 style),")
    print("which is converted to this format by PerturbationPromptFormatterFromJSONL.\n")
    print("Sample Formatted Input:")
    print(formatted_train_ds[0]["model_input"])
    print("\nSample Formatted Response:")
    print(formatted_train_ds[0]["response"])
    
    # Also show original JSONL format for comparison
    if len(train_data) > 0:
        print("\n=== Original JSONL Format (for reference) ===")
        sample_jsonl = train_data[0]
        if "conversations" in sample_jsonl:
            print("Original conversations format:")
            for conv in sample_jsonl["conversations"]:
                print(f"  {conv.get('from', 'unknown')}: {conv.get('value', '')}")
        print("=" * 50 + "\n")
    
    # Load pretrained model
    model_save_dir = os.path.join(output_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    
    csmodel = cs.CSModel(
        model_name_or_path=model_name_or_path,
        save_dir=model_save_dir,
        save_name="perturbation_model"
    )
    print(f"Loaded model: {csmodel}")
    
    # Set max_length on tokenizer
    if max_length and hasattr(csmodel.tokenizer, 'model_max_length'):
        original_max_length = csmodel.tokenizer.model_max_length
        csmodel.tokenizer.model_max_length = max_length
        print(f"Set tokenizer max_length: {original_max_length} -> {max_length}")
    
    # Create training output directory
    timestamp = datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    training_output_dir = os.path.join(output_dir, f"{timestamp}_finetune_{task_name}")
    os.makedirs(training_output_dir, exist_ok=True)
    
    # Define training arguments
    # Check CUDA_VISIBLE_DEVICES to determine GPU setup
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        num_gpus = len([x for x in visible_devices.split(',') if x.strip()])
        print(f"CUDA_VISIBLE_DEVICES={visible_devices}, detected {num_gpus} GPU(s)")
    else:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"No CUDA_VISIBLE_DEVICES set, using all {num_gpus} available GPU(s)")
    
    train_args = TrainingArguments(
        output_dir=training_output_dir,
        run_name=run_name,
        overwrite_output_dir=overwrite_output_dir,
        do_train=do_train,
        do_eval=do_eval,
        # Model & Precision
        bf16=bf16 if torch.cuda.is_available() else False,
        fp16=fp16,
        # Batch size & Accumulation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Optimizer
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        # Training steps
        num_train_epochs=epochs,
        max_steps=max_steps if max_steps is not None else -1,
        # Logging
        logging_strategy=logging_strategy,
        logging_steps=logging_steps,
        logging_first_step=logging_first_step,
        report_to=report_to,
        log_on_each_node=log_on_each_node,
        # Evaluation
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        # Checkpointing
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # Data loading
        dataloader_num_workers=dataloader_num_workers,
        # DDP settings
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_backend="nccl",
        ddp_timeout=ddp_timeout,
        # Memory optimization
        gradient_checkpointing=gradient_checkpointing,
        # DeepSpeed
        deepspeed=deepspeed,
        # Note: model_max_length is set on tokenizer, not TrainingArguments
    )
    
    # Save data split indices
    with open(os.path.join(training_output_dir, 'data_split_indices_dict.pkl'), 'wb') as f:
        pickle.dump(data_split_indices_dict, f)
    
    print(f"\nStarting training...")
    print(f"Output directory: {training_output_dir}")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    # Fine-tune the model
    # Pass the custom prompt_formatter to avoid default C2SPromptFormatter creation
    # (which requires task to be in SUPPORTED_TASKS)
    csmodel.fine_tune(
        csdata=csdata,
        task=task_name,
        train_args=train_args,
        loss_on_response_only=True,
        top_k_genes=num_genes,
        prompt_formatter=prompt_formatter,  # Use custom formatter instead of None
        formatted_hf_ds=formatted_all_ds,
        data_split_indices_dict=data_split_indices_dict,
        num_proc=3,
    )
    
    print(f"\nTraining completed!")
    print(f"Model saved to: {training_output_dir}")
    
    return csmodel, training_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Cell2Sentence model for perturbation response prediction"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file with perturbation data"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="vandijklab/C2S-Scale-Pythia-1b-pt",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name for logging"
    )
    
    # Training parameters
    parser.add_argument("--num_genes", type=int, default=500, help="Number of genes to use")
    parser.add_argument("--num_train_epochs", dest="epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", dest="batch_size", type=int, default=1, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None, help="Evaluation batch size (defaults to train batch size)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (-1 for epochs)")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--logging_first_step", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    
    # Optimization flags
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == 'true', default=True, help="Use bfloat16")
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == 'true', default=False, help="Use float16")
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--ddp_find_unused_parameters", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--ddp_timeout", type=int, default=30000)
    parser.add_argument("--log_on_each_node", type=lambda x: str(x).lower() == 'true', default=False)
    
    # Evaluation flags
    parser.add_argument("--do_train", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--do_eval", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--load_best_model_at_end", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", type=lambda x: str(x).lower() == 'true', default=False)
    
    # Logging
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--overwrite_output_dir", type=lambda x: str(x).lower() == 'true', default=True)
    
    # DeepSpeed
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (auto-set by DeepSpeed)")
    
    # Optimization libraries
    parser.add_argument("--use_liger_kernel", type=lambda x: str(x).lower() == 'true', default=False, help="Use Liger kernel for optimization")
    
    args = parser.parse_args()
    
    # Train the model
    train_perturbation_model(
        input_file=args.input_file,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_genes=args.num_genes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_length=args.max_length,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        logging_first_step=args.logging_first_step,
        logging_strategy=args.logging_strategy,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        val_split=args.val_split,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        ddp_timeout=args.ddp_timeout,
        log_on_each_node=args.log_on_each_node,
        do_train=args.do_train,
        do_eval=args.do_eval,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        overwrite_output_dir=args.overwrite_output_dir,
        deepspeed=args.deepspeed,
        use_liger_kernel=args.use_liger_kernel,
    )


if __name__ == "__main__":
    main()

