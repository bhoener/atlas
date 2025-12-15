import torch
from src.polynomial_mapping import self_tensor, phi_star, batched_phi_star


def test_self_tensor() -> None:
    assert torch.allclose(
        self_tensor(torch.Tensor([0, 1]), 2), torch.Tensor([0, 0, 0, 1])
    )
    assert torch.allclose(
        self_tensor(torch.Tensor([2, 3, -3]), 2),
        torch.Tensor([4, 6, -6, 6, 9, -9, -6, -9, 9]),
    )


def test_phi_star() -> None:
    assert torch.allclose(phi_star(torch.tensor([3]), 0), torch.tensor([1]).float())
    assert torch.allclose(
        phi_star(torch.tensor([4, 2]), 1), torch.tensor([1, 4, 2]).float()
    )
    assert torch.allclose(
        phi_star(torch.tensor([2, 3, -3]), 2),
        torch.cat(
            (
                torch.tensor([1, 2, 3, -3]),
                torch.Tensor([4, 6, -6, 6, 9, -9, -6, -9, 9]).float() / (2**0.5),
            )
        ),
    )


def test_batched_phi_star() -> None:
    assert batched_phi_star(torch.randn(2, 12, 8), 1).size() == (2, 12, 9)
