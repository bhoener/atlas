from src.linear_transformer import poly_sketch_non_negative, fast_lt_mul
import torch

b, k, m, r, p = (4, 8, 4, 6, 4)
B, L, H, h = (4, 128, 8, 4)
Q, K, V = torch.randn(3, B, L, H, h)


def test_normal_lt_mul() -> None:
    qk_similarity_v = (
        torch.tril(
            poly_sketch_non_negative(Q, r, p)
            @ poly_sketch_non_negative(K, r, p).transpose(1, 2)
        )
        @ V
    )
    assert qk_similarity_v.size() == ()


qk_similarity_fast = fast_lt_mul(
    poly_sketch_non_negative(Q, r, p),
    poly_sketch_non_negative(K, r, p),
    V,
)
