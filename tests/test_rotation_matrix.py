import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import numpy.testing as npt

# import einops

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder
from models.torch_models.local_encoder import LocalEncoder as TorchLocalEncoder
from models.equinox_models.local_encoder import LocalEncoder as EquinoxLocalEncoder


def test_compute_rotation_matrix():
    # Setup
    key = jax.random.PRNGKey(0)
    encoder = EquinoxAAEncoder(
        historical_steps=2,
        node_dim=2,
        edge_dim=2,
        embed_dim=2,
        num_heads=2,
        dropout=0.1,
        key=key,
    )
    
    # Test cases
    test_cases = [
        # dpositions, idx, expected rotation matrix
        (jnp.array([[1.0, 0.0]]), 0,  # vector pointing right
         jnp.array([[1.0, 0.0], [0.0, 1.0]])),  # identity matrix (no rotation)
        
        (jnp.array([[0.0, 1.0]]), 0,  # vector pointing up
         jnp.array([[0.0, -1.0], [1.0, 0.0]])),  # 90 degree rotation
        
        (jnp.array([[1.0, 1.0]]), 0,  # 45 degree vector
         jnp.array([[0.7071068, -0.7071068], [0.7071068, 0.7071068]])),  # 45 degree rotation
        
        (jnp.array([[-1.0, 0.0]]), 0,  # vector pointing left
         jnp.array([[-1.0, 0.0], [0.0, -1.0]])),  # 180 degree rotation
    ]
    
    for dpositions, idx, expected in test_cases:
        idx = jnp.array(idx)
        rotate_mat = encoder.compute_rotation_matrix(idx, dpositions)
        
        print(f"\nTest case: {dpositions[idx]}")
        print(f"Computed matrix:\n{rotate_mat}")
        print(f"Expected matrix:\n{expected}")
        
        # Verify matrix properties
        # 1. Shape should be (2, 2)
        assert rotate_mat.shape == (2, 2)
        
        # 2. Should be close to expected matrix
        npt.assert_allclose(rotate_mat, expected, atol=1e-6)
        
        # 3. Should be orthogonal (R @ R.T = I)
        identity = jnp.eye(2)
        npt.assert_allclose(rotate_mat @ rotate_mat.T, identity, atol=1e-6)
        
        # 4. Determinant should be 1 (proper rotation)
        npt.assert_allclose(jnp.linalg.det(rotate_mat), 1.0, atol=1e-6)
        
        # 5. Should rotate vector as expected
        rotated = dpositions[idx] @ rotate_mat
        print(f"Original vector: {dpositions[idx]}")
        print(f"Rotated vector: {rotated}")





if __name__ == "__main__":
    test_compute_rotation_matrix()