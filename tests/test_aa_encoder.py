import jax
import torch
import jax.numpy as jnp
import equinox as eqx
# import einops

from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder

def test_aa_encoder():
    # Set random seed
    key = jax.random.PRNGKey(0)
    torch.manual_seed(0)

    
    # Test parameters
    historical_steps = 3
    node_dim = 4
    edge_dim = 8
    embed_dim = 12
    num_heads = 2
    historical_steps = 3
    dropout = 0.1

        # Initialize model
    eqx_model = EquinoxAAEncoder(
        historical_steps=historical_steps,
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        key=key
    )

        # Initialize PyTorch model
    torch_model = TorchAAEncoder(
        historical_steps=historical_steps,
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        parallel=False
    )


        # Basic assertions to verify initialization
    assert isinstance(eqx_model, EquinoxAAEncoder)
    assert eqx_model.embed_dim == embed_dim
    assert eqx_model.num_heads == num_heads
    assert eqx_model.historical_steps == historical_steps
    assert eqx_model.dropout == dropout
    print("✓ AAEncoder initialization test passed")

    # Basic assertions for PyTorch model
    assert isinstance(torch_model, TorchAAEncoder)
    assert torch_model.embed_dim == embed_dim
    assert torch_model.num_heads == num_heads
    assert torch_model.historical_steps == historical_steps
    print("✓ PyTorch AAEncoder initialization test passed")




if __name__ == "__main__":
    test_aa_encoder()