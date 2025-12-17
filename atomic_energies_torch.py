import torch
from typing import Optional, Union

from mlip.data.helpers.atomic_number_table import AtomicNumberTable


def get_atomic_energies(
    dataset_info,
    atomic_energies_input: Optional[Union[str, dict[int, float]]] = None,
    num_species: Optional[int] = None,
) -> torch.Tensor:
    if num_species is None:
        num_species = len(dataset_info.atomic_energies_map)

    z_table = AtomicNumberTable(sorted(dataset_info.atomic_energies_map.keys()))

    if atomic_energies_input == "average" or atomic_energies_input is None:
        atomic_energies_dict = {
            z_table.z_to_index(z): energy
            for z, energy in dataset_info.atomic_energies_map.items()
        }
        atomic_energies = torch.tensor(
            [atomic_energies_dict[i] for i in range(len(z_table.zs))],
            dtype=torch.float32,
        )

    elif atomic_energies_input == "zero":
        atomic_energies = torch.zeros(num_species, dtype=torch.float32)

    elif isinstance(atomic_energies_input, dict):
        atomic_energies = torch.tensor(
            [atomic_energies_input.get(z, 0.0) for z in range(num_species)],
            dtype=torch.float32,
        )

    else:
        raise ValueError(
            f"The requested strategy for atomic energies "
            f"handling '{atomic_energies_input}' is not supported."
        )

    if len(z_table.zs) > num_species:
        raise ValueError(
            f"len(z_table.zs)={len(z_table.zs)} > num_species={num_species}"
        )

    return atomic_energies
