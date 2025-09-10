from glob import glob

from datasets import load_dataset


files = glob("mask_*.jsonl", root_dir="data/")
data_files = {file[:-6]: file for file in files}
ds = load_dataset("data/", data_files=data_files)
ds.push_to_hub("Ki-Seki/MaskedIO", private=True)
