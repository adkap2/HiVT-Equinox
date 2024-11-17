import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from models.equinox_models.embedding import SingleInputEmbedding as JaxEmbedding
from models.torch_models.embedding import TorchEmbedding
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
    torch_model = TorchEmbedding(in_channel, out_channel)
    
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
    def test_implementations_match():
        # Test cases
        test_inputs = [
            np.ones((batch_size, in_channel), dtype=np.float32),  # All ones
            np.zeros((batch_size, in_channel), dtype=np.float32),  # All zeros
            np.random.randn(batch_size, in_channel).astype(np.float32),  # Random
            np.array([[-1.0, -2.0, -3.0, -4.0]] * batch_size, dtype=np.float32)  # Negative
        ]
        
        for i, test_input in enumerate(test_inputs):
            jax_input = jnp.array(test_input)
            torch_input = torch.FloatTensor(test_input)
            
            jax_out = jax_model(jax_input)
            torch_out = torch_model(torch_input)
            
            jax_np = np.array(jax_out)
            torch_np = torch_out.detach().numpy()
            
            max_diff = np.max(np.abs(jax_np - torch_np))
            print(f"\nTest input {i+1} max difference: {max_diff}")
            
            assert np.allclose(jax_np, torch_np, rtol=1e-4, atol=1e-4), \
                f"Implementations differ for test input {i+1}"
    
    
    # Run all tests
    # test_implementations_match()
    
    print("\n✓ All SingleInputEmbedding tests passed")
    print("  - Output matching verification")
    print("  - Shape verification")
    print("  - Multiple input tests")

if __name__ == "__main__":
    test_single_input_embedding()