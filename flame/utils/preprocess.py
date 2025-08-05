# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
from typing import Any, Dict, List

from transformers import AutoTokenizer, PreTrainedTokenizer

from flame.data import build_dataset
from torchtitan.tools.logging import init_logger, logger

import os
import json

def save_streaming_dataset(dataset, path: str):
    os.makedirs(path, exist_ok=True)
    output_file = os.path.join(path, "data.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info(f"Streaming dataset saved to {output_file}")

def tokenize(
    examples: Dict[str, List[Any]],
    tokenizer: PreTrainedTokenizer,
) -> Dict:
    if 'text' in examples:
        samples = examples['text']
    elif 'content' in examples:
        samples = examples['content']
    else:
        raise ValueError(f'No "text" or "content" field found in examples:\n{examples}')
    input_ids = tokenizer(samples)['input_ids']
    bits_per_token = [len(sample.encode(encoding='utf-8')) * 8 / len(input_ids[i]) for i, sample in enumerate(samples)]
    return {'input_ids': input_ids, 'bits_per_token': bits_per_token}


def get_columns_to_remove(dataset, streaming=False):
    """
    Safely determine which columns to remove from the dataset.
    Handles schema mismatches by only removing non-essential columns.
    """
    try:
        # Try to get the first sample to determine columns
        first_sample = next(iter(dataset))
        all_columns = list(first_sample.keys())
        
        # Keep only essential columns for tokenization (text or content)
        # Remove everything else to avoid schema conflicts
        essential_columns = {'text', 'content'}
        columns_to_remove = [col for col in all_columns if col not in essential_columns]
        
        logger.info(f"Dataset columns: {all_columns}")
        logger.info(f"Columns to remove: {columns_to_remove}")
        
        return columns_to_remove
        
    except Exception as e:
        logger.warning(f"Failed to determine columns dynamically: {e}")
        
        # Fallback: common problematic columns to remove
        # This handles cases where datasets have inconsistent schemas
        fallback_columns = ['meta', 'metadata', 'source', 'url', 'timestamp']
        logger.info(f"Using fallback column removal: {fallback_columns}")
        
        return fallback_columns


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser(description='Preprocess the dataset.')
    parser.add_argument(
        '--dataset',
        default='HuggingFaceFW/fineweb-edu',
        help='Dataset to use, with comma separated values',
    )
    parser.add_argument(
        '--dataset_name',
        default='sample-100BT',
        help='The name of the dataset config, with comma separated values if provided',
    )
    parser.add_argument(
        '--dataset_split',
        default='train',
        help='Dataset split to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='Data dirs to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_files',
        default=None,
        help='Data files to use, with comma separated values if provided',
    )
    parser.add_argument(
        '--data_probs',
        default=None,
        help='Data sampling probabilities, with comma separated values if provided',
    )
    parser.add_argument(
        '--streaming',
        action='store_true',
        help='Whether to use streaming mode',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=64,
        help='Number of workers to use for preprocessing',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for preprocessing',
    )
    parser.add_argument(
        '--path',
        default='data',
        help='Path to save the preprocessed dataset',
    )
    parser.add_argument(
        '--tokenizer',
        default='fla-hub/transformer-1.3B-100B',
        help='Tokenizer to use',
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for processing"
    )
    args = parser.parse_args()

    logger.info(f'Loading tokenizer {args.tokenizer}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    logger.info(f'{tokenizer}')
    logger.info(f'Loading dataset {args.dataset} {args.dataset_name} {args.dataset_split}')
    dataset = build_dataset(
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        data_dir=args.data_dir,
        data_files=args.data_files,
        data_probs=args.data_probs,
        streaming=args.streaming,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # Get columns to remove using the new safe method
    columns_to_remove = get_columns_to_remove(dataset, streaming=args.streaming)
    
    # logger.info(f'Tokenizing and processing the dataset with batch size {args.batch_size}')
    
    # if args.streaming:
    #     # Cannot use remove_columns or save_to_disk with streaming datasets
    #     dataset = dataset.map(
    #         lambda examples: tokenize(examples, tokenizer),
    #         batched=True,
    #         batch_size=args.batch_size,
    #     )
    #     logger.info(f'{dataset}')
    #     logger.info(f'Saving streaming tokenized dataset to {args.path}')
    #     save_streaming_dataset(dataset, args.path)

    # else:
    #     # Get columns to remove using the safe method
    #     columns_to_remove = get_columns_to_remove(dataset, streaming=False)

    #     dataset = dataset.map(
    #         lambda examples: tokenize(examples, tokenizer),
    #         batched=True,
    #         batch_size=args.batch_size,
    #         remove_columns=columns_to_remove,
    #         num_proc=args.num_workers,
    #     )
    #     logger.info(f'{dataset}')
    #     logger.info(f'Saving tokenized dataset to {args.path}')
    #     dataset.save_to_disk(args.path)

    logger.info(f'Tokenizing and processing the dataset with batch size {args.batch_size}')
    
    if args.streaming:
        # Streaming datasets can't use remove_columns or save_to_disk
        tokenized_iterable = dataset.map(
            lambda examples: tokenize(examples, tokenizer),
            batched=True,
            batch_size=args.batch_size,
        )

        os.makedirs(args.path, exist_ok=True)
        output_file = os.path.join(args.path, "data.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for example in tokenized_iterable:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        logger.info(f"Streaming tokenized dataset saved to {output_file}")

    else:
        # Regular map-style dataset logic
        columns_to_remove = get_columns_to_remove(dataset, streaming=False)
        dataset = dataset.map(
            lambda examples: tokenize(examples, tokenizer),
            batched=True,
            batch_size=args.batch_size,
            remove_columns=columns_to_remove,
            num_proc=args.num_workers,
        )
        logger.info(f'{dataset}')
        logger.info(f'Saving tokenized dataset to {args.path}')
        dataset.save_to_disk(args.path)
