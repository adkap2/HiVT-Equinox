import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np

# import einops

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder
from models.torch_models.local_encoder import LocalEncoder as TorchLocalEncoder


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

    # num_nodes: int = 6
    num_nodes: int = 2

    # Create rotation angles (one per actor)
    rotate_angles = torch.tensor([np.pi/4, -np.pi/6])  # example angles: 45° and -30°

    # Create empty rotation matrix
    rotate_mat = torch.empty(num_nodes, 2, 2)  # [2, 2, 2]

    # Calculate sin and cos values
    sin_vals = torch.sin(rotate_angles)  # [2]
    cos_vals = torch.cos(rotate_angles)  # [2]

    # Fill rotation matrix
    rotate_mat[:, 0, 0] = cos_vals    # R_00 = cos(θ)
    rotate_mat[:, 0, 1] = -sin_vals   # R_01 = -sin(θ)
    rotate_mat[:, 1, 0] = sin_vals    # R_10 = sin(θ)
    rotate_mat[:, 1, 1] = cos_vals    # R_11 = cos(θ)


    print("Rotation Matrix shape:", rotate_mat.shape)
    print("Rotation Matrix for first actor:\n", rotate_mat[0])
    print("Rotation Matrix for second actor:\n", rotate_mat[1])

    breakpoint()


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
    # x_torch = torch.randn(batch_size, node_dim) # [2, 2] # one time step
    # Test forward pass
    # x_torch = torch.randn(batch_size, node_dim) # [2, 2]
    x_torch = torch.zeros(batch_size, historical_steps, node_dim)
    for t in range(historical_steps):
        x_torch[0, t] = torch.tensor([t * 0.1, t * 0.1])  # diagonal movement

    for t in range(historical_steps):
        theta = t * np.pi / 20  # circular trajectory
        x_torch[1, t] = torch.tensor([np.cos(theta), np.sin(theta)])


    print("Input tensor shape:", x_torch.shape)
    print("First actor's trajectory:\n", x_torch[0])
    print("Second actor's trajectory:\n", x_torch[1])


    edge_index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) # [2, 2]
    # edge_attr_torch = torch.randn(edge_index_torch.shape[1], edge_dim)
    # edge_attr_torch = torch.randn(
    #     edge_index_torch.shape[1], edge_dim
    # )  # Shape: [num_edges, embed_dim] # [2, 2]

    
    # Create positions for 2 nodes over 1 timestep
    positions = torch.tensor([[[0.0, 0.0],     # Node 0 position
                             [1.0, 1.0]]])     # Node 1 position
    
    # Calculate edge attributes (relative positions)
    edge_attr_torch = positions[0][edge_index_torch[0]] - positions[0][edge_index_torch[1]] # [2, 2]
    

    # Fix: Create valid edge indices (must be within range of number of nodes)
    # edge_index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Only connect nodes 0 and 1
    # edge_attr_torch = torch.randn(edge_index_torch.shape[1], edge_dim)  # Shape: [num_edges, edge_dim]

    bos_mask_torch: torch.Tensor = torch.zeros(batch_size, dtype=torch.bool) # [2]

    # Create identical rotation matrices for both frameworks
    # np_rot_mat: np.ndarray = np.random.randn(batch_size, embed_dim, embed_dim).astype(np.float32) # [2, 2, 2]
    np_rot_mat: np.ndarray = rotate_mat.numpy() # [2, 2, 2]

    # Convert to respective frameworks
    batch_rot_mat_torch: torch.Tensor = torch.tensor(np_rot_mat) # [2, 2, 2]


    torch_output = torch_model(
        x_torch[:, 0], # Shape [batch_size, node_dim] -> [2,2]
        t=0,
        edge_index=edge_index_torch, # Shape: [2, num_edges] -> [2, 2]
        edge_attr=edge_attr_torch, # Shape: [num_edges, edge_dim] -> [2, 2]
        bos_mask=bos_mask_torch, # Shape: [batch_size] -> [2]
        rotate_mat=batch_rot_mat_torch, # Shape: [batch_size, embed_dim, embed_dim] -> [2, 2, 2]
    ) # Shape: [batch_size, embed_dim] -> [2, 2]
    # # Create identical input data
    # Just focus on first actor
    x_eqx = x_torch[:,0].numpy() # Shape: [batch_size, node_dim] -> [2, 2]
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
    print(f"[JAX] eqx_output shape: {eqx_output.shape}")
    print(f"[JAX] eqx_output first few values: {eqx_output[0, :10]}")

    print(f"[TORCH] torch_output shape: {torch_output.shape}")
    print(f"[TORCH] torch_output first few values: {torch_output[0, :10]}")

    breakpoint()


if __name__ == "__main__":
    test_aa_encoder()
