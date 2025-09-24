from pathlib import Path

from datasets import load_dataset


dataset_dir = Path("data/GIM-SFT")
ds_paths = Path.glob(dataset_dir, "*/*.jsonl")

for ds_path in ds_paths:
    print(f"Found data: {ds_path}")
    subset_name = ds_path.parent.stem
    split_name = ds_path.stem

    ds = load_dataset("json", data_files={split_name: ds_path.as_posix()})
    ds.push_to_hub("Ki-Seki/GIM-SFT", subset_name, private=True)
