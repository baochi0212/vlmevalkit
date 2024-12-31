from vlmeval.utils.dataset_config import *
import os
for value in dataset_URLs.values():
    tsv_file = value.split("/")[-1]
    print("File: ", tsv_file)
    url = f"https://huggingface.co/ambivalent02/openencompass_tsv/resolve/main/{tsv_file}?download=true"
    os.system(f"wget {url} -P ./LMUData")
