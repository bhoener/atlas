from __future__ import annotations
import torch
from torch.optim.optimizer import ParamsT


def newtonschulz5(
    G_t: torch.Tensor,
    steps: int = 5,
    epsilon: float = 1e-7,
    constants: tuple[int] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    a, b, c = constants
    X = G_t / (G_t.norm() + epsilon)

    for _ in range(steps):
        X = a * X + b * (X @ X.T @ X) + c * (X @ X.T @ X @ X.T) @ X

    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, parameters: ParamsT, lr: float = 3e-4, momentum: float = 0.9):
        defaults = {
            "lr": lr,
            "momentum": momentum,
        }
        super().__init__(parameters, defaults)

    def step(self) -> None:
        for param_group in self.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if len(state) == 0:
                        state["B"] = torch.zeros_like(p)
                    state["B"] = param_group["momentum"] * state["B"] + p.grad
                    O_t = newtonschulz5(state["B"])
                    p.data -= O_t * param_group["lr"]
