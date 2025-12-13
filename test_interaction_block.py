import torch
import jax
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn


from interaction_block_torch import InteractionBlockTorch
from message_passing_torch import MessagePassingConvolutionTorch
from radial_embedding_torch import (
    polynomial_envelope_updated as torch_poly,
    bessel_basis as torch_bessel,
)

from src.mlip.models.mace.blocks import InteractionBlock
from src.mlip.models.radial_embedding import (
    polynomial_envelope_updated as jax_poly,
    bessel_basis as jax_bessel,
)

# ---------------- data ----------------
np.random.seed(0)

n_nodes = 5
n_edges = 8
radial_dim = 4

edge_vectors_np = np.random.randn(n_edges, 3).astype(np.float32)
jax_edge_vectors = e3nn.IrrepsArray("1o", jnp.array(edge_vectors_np))

edge_lengths_np = np.linalg.norm(edge_vectors_np, axis=1).astype(np.float32)

senders_np = np.random.randint(0, n_nodes, size=n_edges)
receivers_np = np.random.randint(0, n_nodes, size=n_edges)

node_feats_np = np.random.randn(n_nodes, radial_dim).astype(np.float32)
jax_node_feats = e3nn.IrrepsArray("4x0e", jnp.array(node_feats_np))


# ---------------- JAX ----------------
jax_block = InteractionBlock(
    target_irreps="4x0e",
    avg_num_neighbors=3.0,
    l_max=1,
    activation=jax.nn.silu,
)

rng = jax.random.PRNGKey(0)
variables = jax_block.init(
    rng,
    jax_edge_vectors,
    jax_node_feats,
    jax_bessel(jnp.array(edge_lengths_np), 5.0, radial_dim),
    jnp.array(senders_np),
    jnp.array(receivers_np),
)

jax_out = jax_block.apply(
    variables,
    jax_edge_vectors,
    jax_node_feats,
    jax_bessel(jnp.array(edge_lengths_np), 5.0, radial_dim),
    jnp.array(senders_np),
    jnp.array(receivers_np),
).array

# ---------------- TORCH ----------------
torch_block = InteractionBlockTorch(
    target_irreps="4x0e",
    avg_num_neighbors=3.0,
    l_max=1,
    activation=torch.nn.SiLU,
)

torch_out = torch_block(
    torch.tensor(edge_vectors_np),
    torch.tensor(node_feats_np),
    torch_bessel(torch.tensor(edge_lengths_np), 5.0, radial_dim),
    torch.tensor(senders_np),
    torch.tensor(receivers_np),
)

print(
    "Interaction block close:",
    np.allclose(jax_out, torch_out.detach().numpy(), atol=1e-5)
)
