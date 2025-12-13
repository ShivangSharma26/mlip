# NOTE:
# This InteractionBlockTorch is a structural port of the JAX MACE InteractionBlock.
# Exact numerical parity is blocked by missing equivariant tensor product support.

import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from message_passing_torch import MessagePassingConvolutionTorch


class InteractionBlockTorch(nn.Module):
    def __init__(
        self,
        target_irreps: Irreps,
        avg_num_neighbors: float,
        l_max: int,
        activation,
        species_embedding_dim: int | None = None,
    ):
        super().__init__()
        self.target_irreps = Irreps(target_irreps)

        self.linear_up = nn.Identity()

        self.conv = MessagePassingConvolutionTorch(
            avg_num_neighbors=avg_num_neighbors,
            target_irreps=self.target_irreps,
            l_max=l_max,
            activation=activation,
            species_embedding_dim=species_embedding_dim,
        )

        self.linear_down = nn.Identity()

    def forward(
        self,
        edge_vectors,
        node_feats,
        radial_embeddings,
        senders,
        receivers,
        edge_species_feat=None,
    ):
        node_feats = self.linear_up(node_feats)

        node_feats = self.conv(
            edge_vectors,
            node_feats,
            radial_embeddings,
            senders,
            receivers,
            edge_species_feat,
        )

        node_feats = self.linear_down(node_feats)
        return node_feats
