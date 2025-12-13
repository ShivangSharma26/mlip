# blocks_torch.py

import torch
import torch.nn as nn


class RadialEmbeddingBlockTorch(nn.Module):
    def __init__(
        self,
        r_max: float,
        basis_functions,
        envelope_function,
        num_bessel: int,
        avg_r_min: float | None = None,
    ):
        super().__init__()
        self.r_max = r_max
        self.basis_functions = basis_functions
        self.envelope_function = envelope_function
        self.num_bessel = num_bessel
        self.avg_r_min = avg_r_min

    def forward(self, edge_lengths: torch.Tensor):
        basis = self.basis_functions(
            edge_lengths,
            self.r_max,
            self.num_bessel,
        )
        cutoff = self.envelope_function(edge_lengths, self.r_max)

        embedding = basis * cutoff.unsqueeze(-1)

        embedding = torch.where(
            edge_lengths.unsqueeze(-1) == 0.0,
            torch.zeros_like(embedding),
            embedding,
        )

        return embedding
