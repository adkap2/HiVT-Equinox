
import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import numpy.testing as npt

# import einops

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder
from models.torch_models.local_encoder import LocalEncoder as TorchLocalEncoder
from models.equinox_models.local_encoder import LocalEncoder as EquinoxLocalEncoder




def test_compute_rotation_matrices():
    
    # Test case with 2 nodes
    dpositions = jnp.array([
        [1.0, 0.0],  # 0 degrees rotation
        [0.0, 1.0],  # 90 degrees rotation
    ])

    historical_steps = 2
    node_dim = 2
    edge_dim = 2
    embed_dim = 2
    num_heads = 2
    dropout = 0.1
    key = jax.random.PRNGKey(0)
    
    encoder = EquinoxAAEncoder(
        historical_steps=historical_steps,
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        key=key,
    )
    rotate_mat = encoder.compute_rotation_matrices(dpositions)
    
    # Expected results
    expected_0deg = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    expected_90deg = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    
    assert rotate_mat.shape == (2, 2, 2)
    print("rotate_mat", rotate_mat)
    print("expected_0deg", expected_0deg)
    print("expected_90deg", expected_90deg)
    npt.assert_allclose(rotate_mat[0], expected_0deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[1], expected_90deg, atol=1e-6)

    # Now try more complex cases
    dpositions = jnp.array([
        [1.0, 0.0],  # 0 degrees rotation
        [0.0, 1.0],  # 90 degrees rotation
        [1.0, 1.0],  # 45 degrees rotation
        [-1.0, -1.0],  # -45 degrees rotation
        # 180 degrees rotation
        [-1.0, 0.0],
        [0.0, -1.0],
        # 18 degrees rotation
        [0.9510565, 0.309017],
        [-0.309017, 0.9510565],
    ])
    rotate_mat = encoder.compute_rotation_matrices(dpositions)
    print("rotate_mat", rotate_mat)

    expected_0deg = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    expected_90deg = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    expected_45deg = jnp.array([[0.7071068, -0.7071068], [0.7071068, 0.7071068]])
    expected_m45deg = jnp.array([[0.7071068, 0.7071068], [-0.7071068, 0.7071068]])
    expected_180deg = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    expected_18deg = jnp.array([[0.9510565, 0.309017], [-0.309017, 0.9510565]])

    # print the expected results
    print("expected_0deg", expected_0deg)
    print("expected_90deg", expected_90deg)
    print("expected_45deg", expected_45deg)
    print("expected_m45deg", expected_m45deg)
    print("expected_180deg", expected_180deg)
    print("expected_18deg", expected_18deg)

    npt.assert_allclose(rotate_mat[0], expected_0deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[1], expected_90deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[2], expected_45deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[3], expected_m45deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[4], expected_180deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[5], expected_180deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[6], expected_18deg, atol=1e-6)
    npt.assert_allclose(rotate_mat[7], expected_18deg, atol=1e-6)





if __name__ == "__main__":
    test_compute_rotation_matrices()