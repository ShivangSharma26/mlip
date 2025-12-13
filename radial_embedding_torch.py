import torch
import torch.nn as nn
from e3nn.math import soft_one_hot_linspace
from e3nn.math import bessel


def soft_envelope(
    length, max_length, arg_multiplicator: float = 2.0, value_at_origin: float = 1.2
):
    return soft_one_hot_linspace(
        length,
        start=0.0,
        end=max_length,
        number=1,
        basis="smooth_finite",
        cutoff=True,
    ).squeeze(-1) * value_at_origin


def bessel_basis(length: torch.Tensor, max_length: float, number: int) -> torch.Tensor:
    return bessel(length, number, max_length)




def polynomial_envelope_updated(length: torch.Tensor, max_length: float, p: int = 5):
    x = length
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x / max_length, p)
        + p * (p + 2.0) * torch.pow(x / max_length, p + 1)
        - (p * (p + 1.0) / 2.0) * torch.pow(x / max_length, p + 2)
    )
    return envelope * (x < max_length)

