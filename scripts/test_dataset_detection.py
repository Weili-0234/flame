#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flame.data import _is_save_to_disk_dataset

# Test the dataset detection function
test_paths = [
    "./preprocessed_dataset/tokenized_data",  # This should be detected as save_to_disk
    "./preprocessed_dataset/copy_together_long_data/rp_sub",  # This should also be detected
    "mistralai/Mixtral-8x7B",  # This should NOT be detected (HuggingFace hub)
    "/tmp/nonexistent",  # This should NOT be detected
]

print("Testing save_to_disk dataset detection:")
print("=" * 50)

for path in test_paths:
    is_save_to_disk = _is_save_to_disk_dataset(path)
    print(f"Path: {path}")
    print(f"Is save_to_disk dataset: {is_save_to_disk}")
    print("-" * 30)

# Test the actual dataset loading
print("\nTesting dataset loading:")
print("=" * 50)

try:
    from datasets import load_from_disk
    dataset = load_from_disk("./preprocessed_dataset/tokenized_data")
    print(f"Successfully loaded dataset: {dataset}")
    print(f"Dataset info: {dataset.info}")
    print(f"Dataset splits: {dataset.keys()}")
    if 'train' in dataset:
        print(f"Train split size: {len(dataset['train'])}")
        print(f"Train split features: {dataset['train'].features}")
except Exception as e:
    print(f"Error loading dataset: {e}") 