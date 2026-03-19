import torch
from torchviz import make_dot
from atlas import AtlasLayer

B = 4
L = 16
n = 8
d = 64
d_k = 48
r=8
p = 2
chunk_size = 4
x = torch.randn(B, L, n*d_k)

layer = AtlasLayer(d_model=d_k*n, n_heads=n, r=r, p=p, chunk_size=chunk_size)

y = layer(x)

make_dot(y.mean(), params=dict(layer.named_parameters()), show_attrs=True, show_saved=True).render("memory", format="svg")