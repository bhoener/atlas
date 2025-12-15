import torch
import torch.nn as nn
import torch.nn.functional as F
from src.muon import Muon


class Memory(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int, lr: float = 0.1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_h = d_h
        self.d_out = d_out
        self.lr = lr

        self.w1 = nn.Parameter(torch.randn(d_in, d_h, requires_grad=True))
        nn.init.orthogonal_(self.w1)
        self.w2 = nn.Parameter(torch.randn(d_h, d_out, requires_grad=True))
        nn.init.orthogonal_(self.w2)

        self.optim = Muon(self.parameters(), lr=lr)

    def create_mask(self, B: int, chunk_size: int, context_window: int) -> torch.Tensor:
        return torch.tril(torch.ones(B, chunk_size).float()) - torch.tril(
            torch.ones(B, chunk_size).float(), diagonal=-context_window
        )

    def forward(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        context_window: int = 8,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        B, L, D = k.size()
        if L > 1:
            for chunk in range(L // chunk_size):
                x = k[:, chunk * chunk_size : (chunk + 1) * chunk_size, :]
                l1preact = x @ self.w1
                l1 = F.tanh(l1preact)
                out = l1 @ self.w2

                diff = v - out
                diff2 = diff**2.0

                diff2_per_token = diff2.sum(
                    dim=-1, keepdim=True
                )  # (B, c, D) -> (B, c) [per-token errors]

                mask = self.create_mask(B, chunk_size, context_window).unsqueeze(
                    -1
                )  # (B, c)

                masked_diff2 = diff2_per_token * mask

                sum = masked_diff2.sum()
                loss = sum / masked_diff2.numel()

                # with torch.no_grad():
                #     d_masked_diff2 = (
                #         torch.ones_like(masked_diff2).float() / masked_diff2.numel()
                #     )
                #     d_diff2 = mask * d_masked_diff2

                #     d_diff = 2 * diff * d_diff2
                #     d_out = -d_diff
                #     # B -> batch size
                #     # c -> chunk size (tokens)
                #     # D -> d_model
                #     # d_h -> mlp hidden dim

                #     # d_out is of shape (B, c, D)

                #     # w2 is of shape (d_h, D)

                #     # l1 is of shape (B, c, d_h)
                #     d_l1 = torch.einsum("BcD, dD -> Bcd", d_out, self.w2)

                #     # d_out is of shape (B, c, D)

                #     # l1 is of shape (B, c, d_h)

                #     # w2 is of shape (d_h, D)
                #     d_w2 = torch.einsum("BcD, Bcd -> dD", d_out, l1)
                #     d_l1preact = (
                #         4 * (l1preact.exp() + (-l1preact).exp()) ** (-2.0) * d_l1
                #     )

                #     # d_l1preact is of size (B, c, d_h)

                #     # x is of size (B, c, D)

                #     # w1 is of size (D, d_h)

                #     d_w1 = torch.einsum("Bcd, BcD -> Dd", d_l1preact, x)

                #     #self.w1.data -= d_w1 * self.lr
                #     #self.w2.data -= d_w2 * self.lr

                self.optim.zero_grad()
                loss.backward(retain_graph=True)
                self.optim.step()
                # print(f"loss: {loss:.4f}")
        return out
