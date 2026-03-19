import torch
import time
from atlas import tensor_phi_star, newtonschulz5, Memory, AtlasLayer, CausalConv1d, AtlasBlock, Atlas


    
def test_tensor_phi_star() -> None:
    shapes = [(8), (2, 4, 8), (2, 16), (1, 5), (4, 8, 32, 32)]
    for shape in shapes:
        ins = torch.randn(shape)
        out = tensor_phi_star(ins, 2)
        assert out.ndim == ins.ndim
        assert out.size()[:-1] == (ins.size()[:-1])
        assert False

def test_newtonschulz5() -> None:
    shapes = [(2, 4), (16, 8), (5, 10), (11, 3)]
    for shape in shapes:
        ins = torch.randn(shape)
        out = newtonschulz5(ins)
        assert out.shape == ins.shape
        
def test_memory() -> None:
    shapes = [(2, 4, 8), (1, 8, 16), (4, 128, 256)]
    head_dims = [2, 4, 8]
    for shape in shapes:
        B, L, D = shape
        for head_dim in head_dims:
            mem = Memory(D, D // head_dim)
            Q, K, V = torch.randn((3,) + shape)
            Q = Q.view(B, L, -1, head_dim).transpose(1, 2)
            K = K.view(B, L, -1, head_dim).transpose(1, 2)
            V = V.view(B, L, -1, head_dim).transpose(1, 2)
            t0 = time.time()
            out = mem(Q, K, V)
            print((time.time() - t0) * 12)
            assert out.shape == V.shape
    assert False
    

def test_causal_conv1d() -> None:
    cc = CausalConv1d(16, 16, 4)
    out = cc(torch.randn(4, 16, 8))
    assert out.size() == (4, 16, 8)
    
    
def test_atlas_layer() -> None:
    shapes = [(2, 4, 8), (1, 8, 16), (4, 128, 256)]
    head_dims = [2, 4, 8]
    for shape in shapes:
        B, L, D = shape
        for head_dim in head_dims:
            layer = AtlasLayer(D, D // head_dim)
            x = torch.randn(B, L, D)
            t0 = time.time()
            out = layer(x)
            print((time.time() - t0) * 12)
            assert out.shape == x.shape
    assert False
def test_atlas_block() -> None:
    block = AtlasBlock(256, 16)
    
    x = torch.randn(4, 128, 256)

    assert(block(x).size() == x.size())

    assert False

def test_atlas() -> None:
    atlas = Atlas(50304, 256, 16, 12) 
    x = torch.randint(0, 50304, (4, 128))
    
    t0 = time.time()
    out = atlas(x)
    print(time.time() - t0)
    assert out.size() == (4, 128, 50304)
    assert False