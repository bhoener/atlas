import torch
import torch.nn.functional as F
from src.muon import Muon

def test_muon():
    ins = torch.randn(5, 10)
    outs = torch.randn(5, 8)
    
    w = torch.randn(10, 8, requires_grad=True)
    with torch.no_grad():
        w.data /= 10 ** 0.5
    lr = 0.01
    
    opt = Muon([w], lr)
    
    for step in range(100):
        prediction = ins @ w
        loss = F.mse_loss(prediction, outs)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    assert loss < 1.0