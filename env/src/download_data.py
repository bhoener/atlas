from datasets import load_dataset
import tiktoken
import numpy as np
from tqdm import tqdm
enc = tiktoken.get_encoding("gpt2")
data = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

SHARD_SIZE = 10_000_000
DATA_PATH = "data/"

eot = 50256

num_shards = 0
running_length = 0
shard = [eot]
for row in tqdm(data["text"]):
    encoded = enc.encode(row)
    running_length += len(encoded)
    shard.extend(encoded + [eot])
    
    if running_length >= SHARD_SIZE:
        np.save(f"{DATA_PATH}shard_{num_shards:04d}.npy", np.asarray(shard, dtype=np.int32))
        num_shards += 1
        shard = [eot]
        running_length = 0
        
np.save(f"{DATA_PATH}shard_{num_shards+1:04d}.npy", np.asarray(shard, dtype=np.int32))