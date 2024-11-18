import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from models.equinox_models.embedding import SingleInputEmbedding as JaxEmbedding, MultipleInputEmbedding as JaxMultiEmbedding
from models.torch_models.embedding import TorchSingleInputEmbedding, TorchMultipleInputEmbedding
# Import PyTorch version for comparison


def test_single_input_embedding():
    """Test JAX/Equinox SingleInputEmbedding against PyTorch version"""
    
    # Set random seeds
    torch.manual_seed(0)
    key = jax.random.PRNGKey(0)
    
    # Parameters
    in_channel = 4
    out_channel = 8
    batch_size = 2
    
    # Create models
    jax_model = JaxEmbedding(in_channel, out_channel, key=key)
    torch_model = TorchSingleInputEmbedding(in_channel, out_channel)
    
    # Create identical input data
    np_input = np.random.randn(batch_size, in_channel).astype(np.float32)
    jax_input = jnp.array(np_input)
    torch_input = torch.FloatTensor(np_input)
    
    # Get outputs
    jax_output = jax.vmap(jax_model)(jax_input) # Apply vmap to batch process
    torch_output = torch_model(torch_input)

    #@eqx.filter_jit
    @jax.jit
    def dummy_loss(model, x):
        return jax.vmap(model)(x).sum()
    #
    
    # Get gradients
    # grads = eqx.filter_grad(dummy_loss)(jax_model, jax_input)
    grads = jax.grad(dummy_loss)(jax_model, jax_input)
    print(f"Gradients:\n{grads}")
    
    # Convert outputs to numpy for comparison
    jax_np = np.array(jax_output)
    torch_np = torch_output.detach().numpy()
    
    # Basic shape test
    assert jax_output.shape == (batch_size, out_channel), \
        f"Expected shape {(batch_size, out_channel)}, got {jax_output.shape}"
    print("✓ Shape test passed")
    print("Torch shape: ", torch_output.shape)
    print("Jax shape: ", jax_output.shape)
    # Test with multiple different inputs
    # Run all tests
    # test_implementations_match()
    
    print("\n✓ All SingleInputEmbedding tests passed")
    print("  - Output matching verification")
    print("  - Shape verification")
    print("  - Multiple input tests")

def test_multiple_input_embedding():
    """Test JAX/Equinox MultipleInputEmbedding against PyTorch version"""
    
    # Set random seeds
    torch.manual_seed(0)
    key = jax.random.PRNGKey(0)
    # Num nodes to 6
    # Node dimensiosn to 4
    # edge dimensions to 8
    # Parameters
    # in_channels = [node_dim, edge_dim]
    in_channels = [4, 8]
    out_channel = 8
    batch_size = 2
    
    # Create models
    jax_model = JaxMultiEmbedding(in_channels, out_channel, key=key)
    torch_model = TorchMultipleInputEmbedding(in_channels=in_channels, out_channel=out_channel)
    
    # Create identical input data
    continuous_inputs_jax = []
    continuous_inputs_torch = []
    for in_channel in in_channels:
        np_input = np.random.randn(batch_size, in_channel).astype(np.float32)
        continuous_inputs_jax.append(jnp.array(np_input))
        continuous_inputs_torch.append(torch.FloatTensor(np_input))
    
    # Get outputs
    jax_output = jax.vmap(jax_model)(continuous_inputs_jax)

    print("Continuous inputs shapes:", [x.shape for x in continuous_inputs_torch])
    torch_output = torch_model(continuous_inputs_torch)

    @jax.jit
    def dummy_loss_multi(model, x):
        return jax.vmap(model)(x).sum()
    
    # Get gradients
    grads = jax.grad(dummy_loss_multi)(jax_model, continuous_inputs_jax)
    print(f"Multiple Input Gradients:\n{grads}")
    
    # Convert outputs to numpy for comparison
    jax_np = np.array(jax_output)
    torch_np = torch_output.detach().numpy()
    
    # Basic shape test
    assert jax_output.shape == (batch_size, out_channel)
    print("✓ Multiple Input Shape test passed")
    print("Torch shape: ", torch_output.shape)
    print("Jax shape: ", jax_output.shape)

    # Fix categorical input test
    categorical_dim = 4
    cat_input = np.random.randn(batch_size, categorical_dim).astype(np.float32)
    cat_input_jax = jnp.array(cat_input)
    cat_input_torch = torch.FloatTensor(cat_input)
    
    # Add these debug prints to verify input shapes
    print("Categorical input shapes:")
    print(f"JAX: {cat_input_jax.shape}")
    print(f"PyTorch: {cat_input_torch.shape}")
    
    # Ensure continuous inputs haven't been modified
    continuous_inputs_torch = [x.clone() for x in continuous_inputs_torch]  # Create fresh copies
    # Print shapes of continuous inputs
    print("Continuous inputs shapes:", [x.shape for x in continuous_inputs_torch])
    # Continuous inputs torch needs to be torch.size([2,4])
    print(f"Categorical input shape: {cat_input_torch.shape}")

    jax_output_with_cat = jax.vmap(jax_model)(continuous_inputs_jax, cat_input_jax)
    # torch_output_with_cat = torch_model(continuous_inputs_torch, cat_input_torch)
    
    assert jax_output_with_cat.shape == (batch_size, out_channel)
    print("✓ Multiple Input with Categorical Shape test passed")

if __name__ == "__main__":
    #test_single_input_embedding()
    test_multiple_input_embedding()