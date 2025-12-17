import jax.numpy as jnp
#for deepchem 

from mlip.data.dataset_info import DatasetInfo
from mlip.models.atomic_energies import get_atomic_energies


def test_atomic_energies_average():
    dataset_info = DatasetInfo(
        atomic_energies_map={1: -13.6, 6: -102.0},
        cutoff_distance_angstrom=5.0,
    )

    energies = get_atomic_energies(dataset_info)

    assert energies.shape[0] == 2
    assert jnp.allclose(energies, jnp.array([-13.6, -102.0]))


def test_atomic_energies_zero():
    dataset_info = DatasetInfo(
        atomic_energies_map={1: -13.6, 6: -102.0},
        cutoff_distance_angstrom=5.0,
    )

    energies = get_atomic_energies(dataset_info, atomic_energies_input="zero")

    assert jnp.allclose(energies, jnp.zeros(2))


if __name__ == "__main__":
    test_atomic_energies_average()
    test_atomic_energies_zero()
    print("Atomic energies tests passed")
