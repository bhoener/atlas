from src.linear_transformer import (
    poly_sketch_non_negative,
    naive_causal_polysketch_attention,
    causal_polysketch_attention,
    PolySketchAttention,
    MLP,
    DecoderBlock,
    PolySketchFormer,
)
import torch

b, k, m, r, p = (4, 8, 4, 6, 4)
B, L, H, h = (4, 128, 8, 4)
D = H * h
Q, K, V = torch.randn(3, B, L, H, h)


def test_deterministic_polysketch() -> None:
    assert torch.allclose(
        poly_sketch_non_negative(Q, r, p, deterministic=True),
        poly_sketch_non_negative(Q, r, p, deterministic=True),
    )


def test_attention_implementations() -> None:
    Q, K, V = torch.randn(3, B, L, H)
    naive = naive_causal_polysketch_attention(Q, K, V, h, r, p, deterministic=True)
    normal = causal_polysketch_attention(Q, K, V, h, r, p, deterministic=True)
    assert torch.allclose(naive, normal, 0.05)


def test_attention_module() -> None:
    attn = PolySketchAttention(d_model=D, n_heads=H)
    assert attn(torch.randn(B, L, D)).size() == (B, L, D)
    
def test_mlp_module() -> None:
    mlp = MLP(d_in=D, d_h=D*4, d_out=D)
    assert mlp(torch.randn(B, L, D)).size() == (B, L, D)
    
def test_decoder_block_module() -> None:
    db = DecoderBlock(d_model=D, n_heads=H)
    assert db(torch.randn(B, L, D)).size() == (B, L, D)
    
def test_poly_sketch_former_module() -> None:
    psf = PolySketchFormer(d_model=D, n_heads=H, n_layers=12, vocab_size=128, seq_len=1024)
    assert psf(torch.randint(0, 128, (B, L))).size() == (B, L, 128)