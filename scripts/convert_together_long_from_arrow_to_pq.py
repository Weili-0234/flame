import os
from multiprocessing import Pool
from datasets import load_from_disk

DATASETS = ["NI", "P3", "arxiv", "pile_sub", "rp_sub", "ul2_oscar"]

def convert_one(args):
    name, root_dir, output_suffix = args
    input_dir = os.path.join(root_dir, name)
    output_dir = input_dir + output_suffix
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{name}] Loading from {input_dir}")
    ds = load_from_disk(input_dir)

    # Save Parquet shards into the output directory
    parquet_path = os.path.join(output_dir, "data.parquet")
    print(f"[{name}] Saving to {parquet_path}")
    ds.to_parquet(parquet_path)

    print(f"[{name}] Done")
    return name

def convert_all(root_dir: str, output_suffix="_parquet"):
    tasks = [(name, root_dir, output_suffix) for name in DATASETS]
    with Pool(processes=min(len(DATASETS), os.cpu_count())) as pool:
        results = pool.map(convert_one, tasks)
    print("Finished:", results)

if __name__ == "__main__":
    root_dir = "/home/ubuntu/ext-mamba-illinois/wenhao-project/weili/tttdr/flame/preprocessed_dataset/copy_together_long_data"
    convert_all(root_dir)
