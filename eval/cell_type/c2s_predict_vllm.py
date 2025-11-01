"""
Cell2Sentence Cell Type Prediction with vLLM

This script performs efficient batch inference using vLLM for cell type prediction
with Cell2Sentence Gemma models.

Usage:
    python c2s_predict_vllm.py \
        --input_file /path/to/conversations.jsonl \
        --output_dir /path/to/save/results \
        --model_path /path/to/C2S-Scale-Gemma-2-2B \
        --dataset_id A013 \
        --batch_size 256 \
        --tensor_parallel_size 2
"""

# CRITICAL: Set environment variable BEFORE importing vllm
# This must be done before any vllm imports to take effect
import os
os.environ['VLLM_USE_V1'] = '0'

import json
import argparse
import re
from typing import List, Dict
from tqdm import tqdm
from datetime import datetime
import logging

from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_conversation_data(input_file: str) -> List[Dict]:
    """Load conversation data from JSONL file."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"[INFO] Loaded {len(data)} conversation items from {input_file}")
    return data


def prepare_prompts_from_conversations(conversation_items: List[Dict], tokenizer) -> List[str]:
    """
    Convert conversation items to text prompts using chat template.
    
    For Cell2Sentence, we only need the user prompt (not the ground truth).
    """
    prompts = []
    
    for item in conversation_items:
        conversations = item.get('conversations', [])
        
        # Convert 'from' field to 'role' for chat template compatibility
        chat_messages = []
        for conv in conversations:
            if conv.get('from') == 'system':
                chat_messages.append({"role": "system", "content": conv.get('value', '')})
            elif conv.get('from') == 'human':
                chat_messages.append({"role": "user", "content": conv.get('value', '')})
            # Skip 'gpt' messages as those are ground truth answers
        
        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
            try:
                prompt = tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(str(prompt) if prompt is not None else "")
            except Exception as e:
                logging.warning(f"Chat template failed: {e}, using simple format")
                # Fallback to simple format
                user_content = chat_messages[-1]["content"] if len(chat_messages) > 0 else ""
                system_content = chat_messages[0]["content"] if len(chat_messages) > 1 else ""
                prompts.append(f"{system_content}\n\nUser: {user_content}\n\nAssistant:")
        else:
            # No chat template, use simple format
            user_content = ""
            system_content = ""
            for msg in chat_messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]
            prompts.append(f"{system_content}\n\nUser: {user_content}\n\nAssistant:")
    
    return prompts


def extract_ground_truth_from_conversation(conversations: List[Dict]) -> str:
    """Extract ground truth answer from conversations."""
    for conv in conversations:
        if conv.get('from') == 'gpt':
            return conv.get('value', '')
    return ""


def extract_question_from_conversation(conversations: List[Dict]) -> str:
    """Extract question from conversations."""
    for conv in conversations:
        if conv.get('from') == 'human':
            return conv.get('value', '')
    return ""


def clean_prediction(prediction: str) -> str:
    """
    Clean model prediction by removing special tokens and trailing punctuation.
    
    Args:
        prediction: Raw model output
        
    Returns:
        Cleaned prediction
    """
    # Remove special tokens
    prediction = prediction.replace("<ctrl100>", "").strip()
    prediction = prediction.replace("<|endoftext|>", "").strip()
    
    # Remove trailing period if exists
    if prediction.endswith('.'):
        prediction = prediction[:-1]
    
    return prediction.strip()


def run_vllm_inference(
    input_file: str,
    output_dir: str,
    model_path: str,
    dataset_id: str = "unknown",
    max_new_tokens: int = 20,
    batch_size: int = 256,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1
):
    """
    Run vLLM inference on conversation data.
    
    Args:
        input_file: Path to input JSONL file with conversations
        output_dir: Directory to save results
        model_path: Path to C2S model
        dataset_id: Dataset identifier
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        temperature: Sampling temperature
        top_p: Top-p sampling
        top_k: Top-k sampling
    """
    # Note: VLLM_USE_V1 environment variable should be set before importing vllm
    # This is already handled at the top of the script
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load conversation data
    conversation_data = load_conversation_data(input_file)
    
    print(f"\n[INFO] Model: {model_path}")
    print(f"[INFO] Dataset ID: {dataset_id}")
    print(f"[INFO] Tensor parallel size: {tensor_parallel_size}")
    print(f"[INFO] GPU memory utilization: {gpu_memory_utilization}")
    print(f"[INFO] Batch size: {batch_size}")
    
    # Initialize vLLM engine
    logging.info("Initializing vLLM engine (V0)...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()
    
    # Set tokenizer properties
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None,
    )
    
    logging.info(f"Sampling parameters: temp={temperature}, top_p={top_p}, top_k={top_k}, max_tokens={max_new_tokens}")
    
    print(f"[INFO] Starting inference on {len(conversation_data)} cells...")
    
    # Prepare all prompts
    all_prompts = prepare_prompts_from_conversations(conversation_data, tokenizer)
    
    results = []
    
    # Batch inference with vLLM
    logging.info("Starting vLLM batch inference...")
    
    all_outputs = []
    for i in tqdm(range(0, len(all_prompts), batch_size), desc="vLLM Inference"):
        batch_prompts = all_prompts[i:i + batch_size]
        try:
            outputs = llm.generate(batch_prompts, sampling_params)
            all_outputs.extend(outputs)
        except Exception as e:
            logging.error(f"Error in batch {i//batch_size}: {e}")
            all_outputs.extend([None] * len(batch_prompts))
    
    # Infer model name from model path
    if "Gemma" in model_path or "gemma" in model_path:
        model_name = "vandijklab/C2S-Scale-Gemma-2-2B"
    elif "Pythia" in model_path or "pythia" in model_path:
        model_name = "vandijklab/C2S-Pythia-410m-cell-type-prediction"
    else:
        model_name = os.path.basename(model_path)
    
    # Process results
    print("[INFO] Processing results...")
    for idx, (item, output) in enumerate(tqdm(zip(conversation_data, all_outputs), total=len(conversation_data))):
        try:
            if output is not None:
                assistant_reply = output.outputs[0].text.strip()
            else:
                assistant_reply = "ERROR: Generation failed"
            
            # For C2S, the output is directly the cell type
            # Extract the prediction (split by the prompt separator if needed)
            if "The cell type corresponding to these genes is:" in assistant_reply:
                predicted_answer = assistant_reply.split("The cell type corresponding to these genes is:")[1].strip()
            else:
                predicted_answer = assistant_reply
            
            # Clean prediction
            predicted_answer = clean_prediction(predicted_answer)
            
            # Get ground truth and question from conversation
            ground_truth = extract_ground_truth_from_conversation(item.get('conversations', []))
            question = extract_question_from_conversation(item.get('conversations', []))
            
            result_item = {
                "model_name": model_name,
                "dataset_id": dataset_id,
                "index": idx,
                "task_type": "cell type",
                "task_variant": "singlecell_openended",
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "full_response": assistant_reply,
                "group": item.get("group", "")
            }
            results.append(result_item)
            
        except Exception as e:
            logging.error(f"Failed to process result for sample {idx}: {e}")
            ground_truth = extract_ground_truth_from_conversation(item.get('conversations', []))
            question = extract_question_from_conversation(item.get('conversations', []))
            
            result_item = {
                "model_name": model_name,
                "dataset_id": dataset_id,
                "index": idx,
                "task_type": "cell type",
                "task_variant": "singlecell_openended",
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": "",
                "full_response": f"ERROR: {str(e)}",
                "group": item.get("group", "")
            }
            results.append(result_item)
    
    # Save results
    output_file = os.path.join(
        output_dir,
        f"singlecell_openended_predictions_vllm_{timestamp}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] Predictions saved to: {output_file}")
    print(f"[INFO] Total predictions: {len(results)}")
    
    return results, output_file


def main():
    parser = argparse.ArgumentParser(
        description="Cell2Sentence Cell Type Prediction with vLLM"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input conversation JSONL file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to C2S model"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="unknown",
        help="Dataset identifier (e.g., A013, D099)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for vLLM inference (default: 256)"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0, default: 0.9)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1, disabled)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    
    run_vllm_inference(
        input_file=args.input_file,
        output_dir=args.output_dir,
        model_path=args.model_path,
        dataset_id=args.dataset_id,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()

