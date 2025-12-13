# NOTE:
# This is a partial PyTorch port of MACE MessagePassingConvolution.
# Full equivariant tensor product parity with e3nn_jax is not yet supported in PyTorch.
# This implementation is intended as a structural and API-compatible scaffold.


import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics


class MessagePassingConvolutionTorch(nn.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: Irreps,
        l_max: int,
        activation,
        species_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = Irreps(target_irreps)
        self.l_max = l_max
        self.activation = activation
        self.species_embedding_dim = species_embedding_dim

        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            activation(),
            nn.Linear(64, 64),
            activation(),
            nn.Linear(64, self.target_irreps.num_irreps),
        )

        if species_embedding_dim is not None:
            self.mlp_species = nn.Sequential(
                nn.Linear(species_embedding_dim * 3, 64),
                activation(),
                nn.Linear(64, 64),
                activation(),
                nn.Linear(64, self.target_irreps.num_irreps),
            )
        else:
            self.mlp_species = None

    def forward(
        self,
        vectors,
        node_feats,
        radial_embedding,
        senders,
        receivers,
        edge_species_feat=None,
    ):
        messages = node_feats[senders]

        sh = spherical_harmonics(
            list(range(1, self.l_max + 1)),
            -vectors,
            normalize=True,
            normalization="component",
        )

        #messages = messages * sh

        mix = self.mlp(radial_embedding)

        if self.mlp_species is not None:
            mix_species = self.mlp_species(edge_species_feat)
            mix = mix * mix_species

        messages = messages * mix.unsqueeze(-1)


        out = torch.zeros_like(node_feats)
        out.index_add_(0, receivers, messages)

        return out / self.avg_num_neighbors
