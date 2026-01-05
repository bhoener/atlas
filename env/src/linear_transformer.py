import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def poly_sketch_with_negativity(
    A: torch.Tensor, r: int, p: int, gen: torch.Generator | None = None
) -> torch.Tensor:
    if p == 1:
        return A
    gen = torch.Generator().manual_seed(42) if gen is None else gen
    M_1 = poly_sketch_with_negativity(A, r, p // 2, gen)
    M_2 = poly_sketch_with_negativity(A, r, p // 2, gen)

    G_1, G_2 = (
        torch.randn(
            A.size()[:-2]
            + (
                A.size(-1),
                r,
            ),
            generator=gen,
        ),
        torch.randn(
            A.size()[:-2]
            + (
                A.size(-1),
                r,
            ),
            generator=gen,
        ),
    )
    return (1 / r) ** 0.5 * ((M_1 @ G_1) * (M_2 @ G_2))


def poly_sketch_non_negative(A: torch.Tensor, r: int, p: int) -> torch.Tensor:
    M = poly_sketch_with_negativity(A, r, p // 2)
    # paper says M⊗2 ∈ R^{k×r^2}
    # and M ∈ R^{k×r}
    # -> (k, r), (k, r) -> (k, r, r) -> (k, r^2)
    return F.layer_norm(torch.einsum("...i, ...j -> ...ij", M, M).view(A.size()[:-1] + (r**2,)), A.size()[:-1] + (r**2,))


def naive_causal_polysketch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    h: int = 4,
    r: int = 8,
    p: int = 4,
) -> torch.Tensor:
    B, L, D = Q.size()
    H = D // h
    Q, K, V = (
        F.layer_norm(Q.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(K.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(V.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
    )
    phi_Q, phi_K = (
        poly_sketch_non_negative(Q, r, p),
        poly_sketch_non_negative(K, r, p),
    )  # (B, L, H, h) -> (B, L, H, r^2)
    qk_similarity = torch.tril(
        phi_Q @ phi_K.permute(0, 1, 3, 2)
    )  # (B, H, L, r^2), (B, H, r^2, L) -> (B, H, L, L)
    D_tilde = 1.0 + qk_similarity @ torch.ones(L, 1)
    # (B, H, L, L), (L, 1) -> (B, H, L, 1)

    return (qk_similarity @ V) / D_tilde  # (B, H, L, L), (B, H, L, h) -> (B, H, L, h)


def causal_polysketch_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    h: int = 4,
    r: int = 8,
    p: int = 4,
) -> torch.Tensor:
    B, L, D = Q.size()
    H = D // h
    Q, K, V = (
        F.layer_norm(Q.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(K.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
        F.layer_norm(V.view(B, L, H, h).transpose(1, 2), (B, H, L, h)),
    )  # (B, H, L, h)

    phi_Q, phi_K = (
        poly_sketch_non_negative(Q, r, p),
        poly_sketch_non_negative(K, r, p),
    )  # (B, H, L, r^2)

    D_tilde = 1.0 + fast_lt_mul(phi_Q, phi_K, torch.ones(B, H, L, 1))
    # (B, H, L, r^2) @ (B, H, r^2 , L) -> (B, H, L, L), (B, H, L, 1) -> (B, H, L, 1)

    return fast_lt_mul(phi_Q, phi_K, V) / D_tilde  # (B, H, L, h)


def fast_lt_mul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    b: int = 16,
) -> torch.Tensor:
    batch, H, L, r_sq = A.size()
    h = C.size(-1)

    t = L // b
    out = torch.empty(batch, H, L, h)

    Z_l = torch.zeros(batch, H, r_sq, h)
    for l in range(t):
        idx = torch.arange(l * b, (l + 1) * b)

        A_l, B_l, C_l = (
            A[:, :, idx],  # (B, H, b, r^2)
            B[:, :, idx],  # (B, H, b, r^2)
            C[:, :, idx],  # (B, H, b, h) or (B, H, b, 1)
        )

        H_l = torch.einsum(
            "BHbr, BHbh -> BHrh", B_l, C_l
        )  # (B, b, H, r^2), (B, b, H, h) or (B, b, 1, 1) -> (B, H, r^2, h)

        P_l = torch.tril(A_l @ B_l.permute(0, 1, 3, 2)) @ C_l

        # (B, H, b, r^2), (B, H, r^2, b) -> (B, H, b, b)
        # (B, H, b, b), (B, H, b, h) -> (B, H, b, h)

        out[:, :, idx, :] = P_l + A_l @ Z_l
        # (B, H, b, r^2), (B, H, r^2, h) -> (B, H, b, h)

        Z_l = Z_l + H_l
    return out


b, k, m, r, p = (4, 8, 4, 64, 4)

(B, L, H, h) = (4, 128, 2, 4)
Q, K, V = torch.randn(3, B, L, H * h)
polysketch_a = poly_sketch_non_negative(Q, r, p)
polysketch_b = poly_sketch_non_negative(Q, r, p)

print(f"MSE diff (r={r}): {F.mse_loss(polysketch_a, polysketch_b).item()}")

naive = naive_causal_polysketch_attention(Q, K, V, h, r, p)
normal = causal_polysketch_attention(Q, K, V, h, r, p)
print(naive.size(), normal.size())

xs, ys = [], []
for r in range(1, 128):
    xs.append(r)
    poly = poly_sketch_non_negative(Q, r, p)
    ys.append(poly.std())

plt.plot(xs, ys)
plt.show()

# _, axarr = plt.subplots(2, 1)
# axarr[0].imshow(naive.reshape(-1, L))
# axarr[0].set_title("Naive Implementation")
# axarr[1].imshow(normal.reshape(-1, L))
# axarr[1].set_title("Fast Lower Triangular Multiplication")
# plt.show()
