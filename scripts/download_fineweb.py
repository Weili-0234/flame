
import os
from datasets import load_dataset

# Set your Hugging Face cache path
os.environ["HF_HOME"] = "/home/ubuntu/ext-mamba-illinois/wenhao-project/enxin/llava-next/hf_cache"
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ["HF_HOME"], "datasets")

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

# from datasets import load_dataset

# # load fineweb-edu with parallel processing
# dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="default", num_proc=32)

# # or load a subset with roughly 100B tokens, suitable for small- or medium-sized experiments
# # dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", num_proc=64, cache_dir="/your/cache/path")

# Subsets you want to download in order
subsets = ["sample-10BT", "sample-100BT", "sample-350BT"]

# Use streaming=False to trigger full download
def download_subset(subset_name):
    print(f"Downloading {subset_name}...")
    _ = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=subset_name,
        split="train",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
        num_proc=8,   # Parallel preprocessing if needed
        streaming=False,
    )
    print(f"Finished downloading {subset_name}.")

if __name__ == "__main__":
    # Download sequentially to honor nested sampling dependencies
    for subset in subsets:
        download_subset(subset)
