import torch
import jax
import jax.numpy as jnp
import numpy as np

from radial_embedding_torch import (
    polynomial_envelope_updated as torch_poly,
    bessel_basis as torch_bessel,
)
from blocks_torch import RadialEmbeddingBlockTorch

from src.mlip.models.blocks import RadialEmbeddingBlock
from src.mlip.models.radial_embedding import (
    polynomial_envelope_updated as jax_poly,
    bessel_basis as jax_bessel,
)

# inputs
edge_lengths_np = np.random.rand(10).astype(np.float32) * 4.0
r_max = 5.0
num_bessel = 4

# JAX block
jax_block = RadialEmbeddingBlock(
    r_max=r_max,
    basis_functions=jax_bessel,
    envelope_function=jax_poly,
    num_bessel=num_bessel,
)

rng = jax.random.PRNGKey(0)
variables = jax_block.init(rng, jnp.array(edge_lengths_np))
jax_out = jax_block.apply(variables, jnp.array(edge_lengths_np)).array

#jax_out = jax_block(jnp.array(edge_lengths_np)).array

# Torch block
torch_block = RadialEmbeddingBlockTorch(
    r_max=r_max,
    basis_functions=torch_bessel,
    envelope_function=torch_poly,
    num_bessel=num_bessel,
)
torch_out = torch_block(torch.tensor(edge_lengths_np))

print(
    "Block close:",
    np.allclose(jax_out, torch_out.detach().numpy(), atol=1e-6)
)
