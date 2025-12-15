import torch
import math


def self_tensor(x: torch.Tensor, p: int) -> torch.Tensor:
    assert p >= 0, "p must be a positive integer"
    if p == 0:
        return torch.tensor([1]).to(x.dtype)
    elif p == 1:
        return x
    return torch.kron(x, self_tensor(x, p - 1))


def phi_star(x: torch.Tensor, p: int) -> torch.Tensor:
    assert x.ndim >= 1, "input must be a vector"
    if p == 0:
        return torch.tensor([1.0])
    return torch.cat(
        [self_tensor(x.float(), n) / math.sqrt(math.factorial(n)) for n in range(p + 1)]
    )


def batched_phi_star(x: torch.Tensor, p: int) -> torch.Tensor:
    B, T, _ = x.size()
    return torch.stack(
        [torch.stack([phi_star(x[b][t], p) for t in range(T)]) for b in range(B)]
    )


def main() -> None:
    ins = torch.randn(2)
    p = 4
    out = phi_star(ins, p)
    print(ins)
    print(ins.shape)
    print(p)
    print(out)
    print(out.shape)


if __name__ == "__main__":
    main()
