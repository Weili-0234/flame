import os
from datasets import load_dataset

HF_FILES = {
    "NI": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/NI_decontaminated_materialized.jsonl.zst",
    "P3": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/P3_decontaminated_materialized.jsonl.zst",
    "arxiv": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/arxiv_doc_to_abs.jsonl.zst",
    "pile_sub": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/pile_sub.jsonl.zst",
    "rp_sub": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/rp_sub.jsonl.zst",
    "ul2_oscar": "https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/pretrain/ul2_plus_oscar_en.jsonl.zst",
}

def process_file(name: str, url: str, output_root: str):
    print(f"\nProcessing {name} from {url}...")
    dataset = load_dataset("json", data_files=url, split="train")

    if "meta" in dataset.column_names:
        print(" - Removing 'meta' column")
        dataset = dataset.remove_columns("meta")

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)
    print(f" - Saved {name} dataset to {out_dir}")

if __name__ == "__main__":
    output_root = "/home/ubuntu/ext-mamba-illinois/wenhao-project/weili/tttdr/flame/preprocessed_dataset/copy_together_long_data"
    os.makedirs(output_root, exist_ok=True)

    for name, url in HF_FILES.items():
        process_file(name, url, output_root)

