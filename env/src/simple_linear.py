# based on https://sustcsonglin.github.io/blog/2024/deltanet-1/

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def view(x: torch.Tensor, shape: tuple[int] | None = None) -> None:
    plt.imshow(x.detach().view(shape if shape is not None else x.size()).cpu().numpy())
    plt.show()


d_head = 16
L = 32

Q = torch.randn(L, d_head)
K = torch.randn(L, d_head)
V = torch.randn(L, d_head)

M = torch.triu(torch.ones(L, L) * float("-inf"), diagonal=1)

outer_product = Q @ K.T  # (L, d) x (d, L) -> (L, L)

masked = outer_product + M

normalized = F.softmax(masked, dim=-1)

out = normalized @ V  # (L, L) x (L, d) -> (L, d)


# iterative inference

for t in range(L):
    o_t = torch.zeros(d_head)

    q_t = torch.randn(d_head)  # (d)
    for j in range(t):
        k_j = K[j]  # (d)
        v_j = V[j]  # (d)

        numerator = (q_t.T @ k_j).exp()  # (1)

        denominator = 0
        for l in range(t):
            denominator += (q_t.T @ k_j).exp()  # (1)

        o_t += (numerator / denominator) * v_j  # (d)

# O = softmax(QK^T * M)V

# if we remove softmax, O = (QK^T * M)V
# (QK^T)V = Q(K^TV)
# let S_t = sum_j=1^t(v_j k_j^T)

# we know that <qk^T, v> = v(q^Tk)
# o_t = sum_j v_j(k_j^T q_t)
# = sum_j (v_j k_j^T)q_t = S_t q_t

# we can have S_t = S_t-1 + v_t k_t^T


# for each timestep t:
#   calculate q_t, k_t, v_t
#   update state: S_t = S_t-1 + v_t k_t^T
#   find o_t = S_t q

S_t = torch.zeros(d_head, d_head)

for t in range(L):
    q_t = Q[t]  # (d)
    k_t = K[t]  # (d)
    v_t = V[t]  # (d)

    S_t = S_t + (v_t @ k_t.T)  # (d, 1) x (1, d) -> (d, d)

    o_t = S_t @ q_t  # (d, d) x (d, 1) -> (d, 1)


class LinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0

        self.d_head = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()

        Q = (
            self.wq(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, L, D) -> (B, L, H, d) -> (B, H, L, d)
        K = (
            self.wk(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, L, D) -> (B, L, H, d) -> (B, H, L, d)
        V = (
            self.wv(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        )  # (B, L, D) -> (B, L, H, d) -> (B, H, L, d)

        S = torch.zeros(B, self.n_heads, self.d_head, self.d_head)  # (B, H, d, d)

        out = torch.zeros(B, L, D)
        for t in range(L):
            q_t = (
                Q[:, :, t, :].unsqueeze(2).transpose(-2, -1)
            )  # (B, H, 1, d) -> (B, H, d, 1)
            k_t = (
                K[:, :, t, :].unsqueeze(2).transpose(-2, -1)
            )  # (B, H, 1, d) -> (B, H, d, 1)
            v_t = (
                V[:, :, t, :].unsqueeze(2).transpose(-2, -1)
            )  # (B, H, 1, d) -> (B, H, d, 1)

            S = S + v_t @ k_t.transpose(-2, -1)  # (B, H, d, d)

            out[:, t, :] = (S @ q_t).view(
                B, D
            )  # (B, H, d, d) x (B, H, d, 1) -> (B, H, d, 1) -> (B, D)

        return self.wo(out)


attn = LinearAttention(32, 4)

view(attn(torch.randn(2, 8, 32)), (2 * 8, 32))
