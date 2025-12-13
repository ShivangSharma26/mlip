import torch
import jax
import jax.numpy as jnp
import numpy as np

from radial_embedding_torch import (
    polynomial_envelope_updated as torch_poly,
    bessel_basis as torch_bessel,
)

from src.mlip.models.radial_embedding import (
    polynomial_envelope_updated as jax_poly,
    bessel_basis as jax_bessel,
)

# fixed seed
torch.manual_seed(0)
key = jax.random.PRNGKey(0)

length_np = np.random.rand(5).astype(np.float32) * 3.0
max_length = 5.0
number = 4

# JAX
jax_len = jnp.array(length_np)
jax_poly_out = jax_poly(jax_len, max_length)
jax_bessel_out = jax_bessel(jax_len, max_length, number)

# Torch
torch_len = torch.tensor(length_np)
torch_poly_out = torch_poly(torch_len, max_length)
torch_bessel_out = torch_bessel(torch_len, max_length, number)

print("Polynomial close:",
      np.allclose(np.array(jax_poly_out), torch_poly_out.detach().numpy(), atol=1e-6))

print("Bessel close:",
      np.allclose(np.array(jax_bessel_out), torch_bessel_out.detach().numpy(), atol=1e-6))
