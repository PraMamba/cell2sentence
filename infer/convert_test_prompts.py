#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert test JSONL prompts and responses to match training format

This script converts both the "human" question prompts and "gpt" responses in test JSONL files 
to match the format used during training (PerturbationPromptFormatterFromJSONL).

- Human prompt: Converted to training format with control cell sentence
- GPT response: Converted to space-separated gene list format (matching answer_template)

Usage:
    python convert_test_prompts.py \
        --input_file /path/to/test.jsonl \
        --output_file /path/to/converted_test.jsonl \
        --num_genes 500
"""

import os
import json
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm


def convert_prompt_format(
    record: Dict,
    num_genes: int = None
) -> Tuple[str, str]:
    """
    Convert a record's prompt and response to match training format.
    
    Args:
        record: JSONL record containing gene_lists and metadata
        num_genes: Number of genes to use (if None, uses all available)
    
    Returns:
        Tuple of (formatted_prompt, formatted_response) matching training format
    """
    # Extract gene lists
    gene_lists = record.get('gene_lists', {})
    pre_genes = gene_lists.get('pre_perturbation_genes', [])
    post_genes = gene_lists.get('post_perturbation_genes', [])
    
    # Extract metadata
    metadata = record.get('metadata', {})
    cell_type = metadata.get('cell_type', 'Unknown')
    treatment = metadata.get('treatment', 'Unknown')
    
    # Determine number of genes to use
    if num_genes is None:
        # Try to get from metadata, otherwise use all available
        num_genes = metadata.get('pre_genes_count', len(pre_genes))
    
    # Limit to top num_genes
    control_genes = pre_genes[:num_genes]
    perturbed_genes = post_genes[:num_genes]
    
    # Format gene lists as space-separated strings
    control_sentence = ' '.join(control_genes)
    perturbed_sentence = ' '.join(perturbed_genes)
    
    # Format prompt according to training template
    prompt = f"""Given the following cell sentence of {len(control_genes)} expressed genes representing a {cell_type} cell's basal state, predict the cell sentence after applying the perturbation: {treatment}.
Control cell sentence: {control_sentence}.

Perturbed cell sentence:"""
    
    # Format response according to training template (space-separated genes with period)
    response = f"{perturbed_sentence}."
    
    return prompt, response


def convert_jsonl_prompts(
    input_file: str,
    output_file: str,
    num_genes: int = None
):
    """
    Convert prompts and responses in JSONL file to match training format.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        num_genes: Number of genes to use (if None, uses metadata.pre_genes_count or all available)
    
    The conversion:
    - Replaces "human" prompts with training format prompt
    - Replaces "gpt" responses with space-separated gene list (matching answer_template)
    - Keeps "system" messages unchanged
    """
    print(f"Loading data from: {input_file}")
    
    converted_records = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        # Count total lines for progress bar
        total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        
        # Reset and process
        for line in tqdm(f_in, total=total_lines, desc="Converting prompts"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                # Validate required fields
                if 'conversations' not in record:
                    print(f"Warning: Record missing 'conversations' field, skipping")
                    continue
                
                if 'gene_lists' not in record:
                    print(f"Warning: Record missing 'gene_lists' field, skipping")
                    continue
                
                if 'metadata' not in record:
                    print(f"Warning: Record missing 'metadata' field, skipping")
                    continue
                
                # Convert the prompt and response
                new_prompt, new_response = convert_prompt_format(record, num_genes)
                
                # Update conversations: replace "human" and "gpt" messages
                conversations = record['conversations'].copy()
                updated_conversations = []
                
                for conv in conversations:
                    if conv.get('from') == 'human':
                        # Replace with new prompt
                        updated_conversations.append({
                            'from': 'human',
                            'value': new_prompt
                        })
                    elif conv.get('from') == 'gpt':
                        # Replace with new response (space-separated gene list)
                        updated_conversations.append({
                            'from': 'gpt',
                            'value': new_response
                        })
                    else:
                        # Keep other messages (system) as-is
                        updated_conversations.append(conv)
                
                # Create new record with updated conversations
                new_record = record.copy()
                new_record['conversations'] = updated_conversations
                
                converted_records.append(new_record)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing record: {e}")
                continue
    
    # Write converted records
    print(f"\nWriting {len(converted_records)} converted records to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in converted_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"[SUCCESS] Conversion completed!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Records: {len(converted_records)}")
    
    # Show sample conversion
    if len(converted_records) > 0:
        print("\n=== Sample Conversion ===")
        sample = converted_records[0]
        for conv in sample['conversations']:
            if conv.get('from') == 'human':
                print("New human prompt (first 200 chars):")
                print(conv.get('value', '')[:200] + "...")
            elif conv.get('from') == 'gpt':
                print("\nNew gpt response (first 200 chars):")
                print(conv.get('value', '')[:200] + "...")
                print(f"Total response length: {len(conv.get('value', ''))} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Convert test JSONL prompts to match training format"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--num_genes",
        type=int,
        default=None,
        help="Number of genes to use (default: use metadata.pre_genes_count or all available)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    convert_jsonl_prompts(
        input_file=args.input_file,
        output_file=args.output_file,
        num_genes=args.num_genes
    )


if __name__ == "__main__":
    main()

