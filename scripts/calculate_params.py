import json
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import sys
import os

# Add the parent directory of 'scripts' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom models to make them available to AutoModelForCausalLM
import fla
import flame
import custom_models

def main():
    parser = argparse.ArgumentParser(description='Calculate the number of parameters for a LaCT LLM model.')
    parser.add_argument('config_path', type=str, help='Path to the model config.json file.')
    args = parser.parse_args()

    try:
        # Load the configuration using AutoConfig
        model_config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)

        # Instantiate the model on the 'meta' device to avoid allocating memory
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)

        # Calculate total trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config_path}'")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print(f"Calculating parameters for: {args.config_path}")
    print("-" * 50)
    print(f"Model Configuration:")
    print(f"  Model Type: {model_config.model_type}")
    print(f"  Hidden Size: {model_config.hidden_size}")
    print(f"  Num Hidden Layers: {model_config.num_hidden_layers}")
    print(f"  Vocab Size: {model_config.vocab_size}")
    if hasattr(model_config, 'num_lact_heads'):
        print(f"  Num LaCT Heads: {model_config.num_lact_heads}")
    if hasattr(model_config, 'w0_w2_low_rank'):
        print(f"  w0_w2_low_rank: {model_config.w0_w2_low_rank}")
    print("-" * 50)
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"                           ~{total_params / 1_000_000:.2f} M")
    print(f"                           ~{total_params / 1_000_000_000:.3f} B")
    print("-" * 50)

if __name__ == '__main__':
    main()