import torch
import torch.nn as nn
from src.memory import Memory

def test_memory():
    B, L, D = (4, 16, 128)
    mem = Memory(D, D*4, D)
    
    wq = nn.Linear(D, D)
    wk = nn.Linear(D, D)
    wv = nn.Linear(D, D)
    
    wo = nn.Linear(D, D)
    opt = torch.optim.AdamW([wq.weight, wq.bias, wk.weight, wk.bias, wv.weight, wv.bias, wo.weight, wo.bias], lr=0.1)
    
    for _ in range(100):
        x = torch.randn(B, L, D)
        Q, K, V = wq(x), wk(x), wv(x)
        v_hat = mem(K, V, 8, 16)
        assert v_hat.shape == V.shape
        assert Q.shape == (B, L, D)
        
        out = wo(v_hat)
        
        loss = out.sum().abs()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
