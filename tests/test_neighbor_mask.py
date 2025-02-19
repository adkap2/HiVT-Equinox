
import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import numpy.testing as npt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# import einops

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder
from models.torch_models.local_encoder import LocalEncoder as TorchLocalEncoder
from models.equinox_models.local_encoder import LocalEncoder as EquinoxLocalEncoder



def test_create_neighbor_mask():
    # Setup
    encoder = EquinoxAAEncoder(
        historical_steps=2,
        node_dim=2,
        edge_dim=2,
        embed_dim=2,
        num_heads=2,
        dropout=0.1,
        key=jax.random.PRNGKey(0),
    )
    positions = jnp.array([
        [0., 0.],  # center node
        [1., 0.],  # close neighbor
        [100., 100.],  # far neighbor
        [10., 10.],  # medium neighbor
        [50., 50.],  # far neighbor
        [30., 30.],  # far neighbor
        [25., 25.],  # far neighbor
        [20., 20.],  # Close but padded
    ])
    padding_mask = jnp.array([False, False, False, False, False, False, False, True])
    bos_mask = jnp.array([False, False, False, False, False, False, False, False])
    idx = jnp.array(0)

    # Test
    mask = encoder.create_neighbor_mask(idx, positions, padding_mask, bos_mask)

        # Print distances for clarity
    distances = jnp.sqrt(jnp.sum((positions - positions[0]) ** 2, axis=1))
    print("Distances from center:", distances)
    print("Mask:", mask)
    print("Positions:", positions)
    print("Padding mask:", padding_mask)
    
    # Verify
    assert mask.shape == (1, 8)
    # Center node should always connect to itself
    assert mask[0, 0] == True
    # Close neighbor should be connected
    assert mask[0, 1] == True
    # Far neighbor should not be connected
    assert mask[0, 2] == False
    # Padded node should not be connected
    assert mask[0, 3] == True

    assert mask[0, 4] == False
    assert mask[0, 5] == True
    assert mask[0, 6] == True
    assert mask[0, 7] == False

    print("mask", mask)
    print("positions", positions)
    print("padding_mask", padding_mask)
    print("bos_mask", bos_mask)


if __name__ == "__main__":
    test_create_neighbor_mask()