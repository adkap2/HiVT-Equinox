import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np

# import einops

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder


def test_aa_encoder():
    # Set random seed
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    torch.manual_seed(0)

    # Test parameters
    batch_size: int = 2
    # batch_size = 32
    historical_steps: int = 20
    node_dim: int = 2
    # node_dim = 8
    edge_dim: int = 2
    embed_dim: int = 2
    # embed_dim = 8
    # num_heads = 8
    num_heads: int = 2
    dropout: float = 0.1

    num_nodes: int = 6

    # Initialize model
    eqx_model = EquinoxAAEncoder(
        historical_steps=historical_steps, # int
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        key=key,
    )

    # Initialize PyTorch model
    torch_model = TorchAAEncoder(
        historical_steps=historical_steps, # int
        node_dim=node_dim, # int
        edge_dim=edge_dim, # int
        embed_dim=embed_dim, # int
        num_heads=num_heads, # int
        dropout=dropout, # float
        parallel=False,
    ) 

    # Basic assertions to verify initialization
    assert isinstance(eqx_model, EquinoxAAEncoder)
    assert eqx_model.embed_dim == embed_dim
    assert eqx_model.num_heads == num_heads
    assert eqx_model.historical_steps == historical_steps
    assert eqx_model.dropout == dropout
    # print("✓ AAEncoder initialization test passed")

    # Basic assertions for PyTorch model
    assert isinstance(torch_model, TorchAAEncoder)
    assert torch_model.embed_dim == embed_dim
    assert torch_model.num_heads == num_heads
    assert torch_model.historical_steps == historical_steps
    # print("✓ PyTorch AAEncoder initialization test passed")

    # Test forward pass
    x_torch = torch.randn(batch_size, node_dim) # [2, 2]
    edge_index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) # [2, 2]
    # edge_attr_torch = torch.randn(edge_index_torch.shape[1], edge_dim)
    edge_attr_torch = torch.randn(
        edge_index_torch.shape[1], edge_dim
    )  # Shape: [num_edges, embed_dim] # [2, 2]

    # Fix: Create valid edge indices (must be within range of number of nodes)
    # edge_index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Only connect nodes 0 and 1
    # edge_attr_torch = torch.randn(edge_index_torch.shape[1], edge_dim)  # Shape: [num_edges, edge_dim]

    bos_mask_torch: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool) # [2]

    # Create identical rotation matrices for both frameworks
    np_rot_mat: np.ndarray = np.random.randn(batch_size, embed_dim, embed_dim).astype(np.float32) # [2, 2, 2]

    # Convert to respective frameworks
    batch_rot_mat_torch: torch.Tensor = torch.tensor(np_rot_mat) # [2, 2, 2]
    torch_output = torch_model(
        x_torch, # Shape [batch_size, node_dim] -> [2,2]
        t=0,
        edge_index=edge_index_torch, # Shape: [2, num_edges] -> [2, 2]
        edge_attr=edge_attr_torch, # Shape: [num_edges, edge_dim] -> [2, 2]
        bos_mask=bos_mask_torch, # Shape: [batch_size] -> [2]
        rotate_mat=batch_rot_mat_torch, # Shape: [batch_size, embed_dim, embed_dim] -> [2, 2, 2]
    ) # Shape: [batch_size, embed_dim] -> [2, 2]
    # Create identical input data
    # np_input: np.ndarray = np.random.randn(batch_size, node_dim).astype(np.float32) # Shape [batch_size, node_dim] -> [2, 2]
    x_eqx = x_torch.numpy()
    jax_input: jnp.ndarray = jnp.array(x_eqx) # Shape: [batch_size, node_dim] -> [2, 2] 

    eqx_batch_rot_mat: jnp.ndarray = jnp.array(batch_rot_mat_torch) # Shape: [batch_size, embed_dim, embed_dim] -> [2, 2, 2]
    bos_mask_jax: jnp.ndarray = jnp.array(bos_mask_torch.numpy()) # Shape: [batch_size] -> [2]

    edge_index_jax = jnp.array(edge_index_torch.numpy()) # [2, 2]
    edge_attr_jax = jnp.array(edge_attr_torch.numpy()) # [2, 2]

    eqx_output: jnp.ndarray = eqx_model(
        jax_input, # Shape: [batch_size, node_dim] -> [2, 2]
        edge_index_jax, # Shape: [2, num_edges] -> [2, 2]
        edge_attr_jax, # Shape: [num_edges, edge_dim] -> [2, 2]
        bos_mask_jax, # Shape: [batch_size] -> [2]
        t=0,
        rotate_mat=eqx_batch_rot_mat, # Shape: [batch_size, embed_dim, embed_dim] -> [2, 2, 2]
    ) # Shape: [batch_size, embed_dim] -> [2, 2]

    # Compare outputs
    # print(f"[JAX] eqx_output shape: {eqx_output.shape}")
    # print(f"[JAX] eqx_output first few values: {eqx_output[0, :5]}")

    # print(f"[TORCH] torch_output shape: {torch_output.shape}")
    # print(f"[TORCH] torch_output first few values: {torch_output[0, :5]}")

    # breakpoint()


if __name__ == "__main__":
    test_aa_encoder()
