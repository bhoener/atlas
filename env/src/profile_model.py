import torch
from atlas import Atlas

from torch.profiler import profile, ProfilerActivity, record_function

vocab_size = 50304
d_model = 256
n_heads = 16
n_layers = 12
batch_size = 2
seq_len = 128

model = Atlas(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers)

device = torch.device("cuda:0")

model = model.to(device)

x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(x)



prof.export_chrome_trace("trace.json")

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))