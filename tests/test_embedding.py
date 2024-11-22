import jax
import jax.numpy as jnp
import numpy as np
import torch
import pytest
from models.equinox_models.embedding import SingleInputEmbedding as JaxEmbedding, MultipleInputEmbedding as JaxMultiEmbedding
from models.torch_models.embedding import TorchSingleInputEmbedding, TorchMultipleInputEmbedding
# Import PyTorch version for comparison
import equinox as eqx
from beartype import beartype
from jaxtyping import Array, Float, PRNGKeyArray
from typing import List, Tuple

# Import bear type
from jaxtyping import Array, Float, PRNGKeyArray
# Bear type


@beartype
def test_single_input_embedding() -> None:
    """Test JAX/Equinox SingleInputEmbedding against PyTorch version"""
    
    # Set random seeds
    torch.manual_seed(0)
    key = jax.random.PRNGKey(0)
    
    # Parameters
    # in_channel = 4 # Node Dim
    # out_channel = 8 # Embed Dim
    node_dim = 2
    embed_dim = 2
    batch_size = 2

    # Test with multiple sizes to ensure the property that multiple input factors are supported
    
    # Create models
    jax_model = JaxEmbedding(node_dim, embed_dim, key=key)

    torch_model = TorchSingleInputEmbedding(node_dim, embed_dim)

    # torch_model.apply(torch.nn.init.ones_)

    # Now init for equinox using the model surgery
    #linear = eqx.nn.Linear(...)
    #new_weight = jax.random.normal(...)
    #where = lambda l: l.weight
    #new_linear = eqx.tree_at(where, linear, new_weight)
    """def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
        out, in_ = weight.shape
        stddev = math.sqrt(1 / in_)
        return stddev * jax.random.truncated_normal(key, shape=(out, in_), lower=-2, upper=2)"""

    """python
    import equinox as eqx
    import jax.tree_util
    import jax.numpy as jnp

    def initialize_params(model, value=1.0):
        #Initialize all parameters in a PyTree model to a specified value.
        def init_fn(x):
            if isinstance(x, (jnp.ndarray, jnp.generic)):
                return jnp.full_like(x, value)
            return x  # Return non-array leaves unchanged

        return tree_util.tree_map(init_fn, model)
    """
    # Force all weights to be the same (1 )
    
    # Create identical input data
    np_input = np.random.randn(batch_size, node_dim).astype(np.float32)
    jax_input = jnp.array(np_input)
    torch_input = torch.FloatTensor(np_input)
    
    # Get outputs

    # jax_output = jax.vmap(jax_model)(jax_input)
    jax_output = jax_model(jax_input)
    # Get outputs

    torch_output = torch_model(torch_input)

    #@eqx.filter_jit
    # @jax.jit
    # @eqx.filter_jit
    # def dummy_loss(model, x):
    #     # return jax.vmap(model)(x).sum()
    #     return model(x).sum()
    # #
    
    # # Get gradients
    # # grads = eqx.filter_grad(dummy_loss)(jax_model, jax_input)
    # grads = jax.grad(dummy_loss)(jax_model, jax_input)
    # print(f"Gradients:\n{grads}")
    
    # Convert outputs to numpy for comparison
    jax_np = np.array(jax_output)
    torch_np = torch_output.detach().numpy()
    
    # Basic shape test
    assert jax_output.shape == (batch_size, embed_dim), \
        f"Expected shape {(batch_size, embed_dim)}, got {jax_output.shape}"
    print("✓ Shape test passed")
    print("Torch shape: ", torch_output.shape)
    print("Jax shape: ", jax_output.shape)
    # Test with multiple different inputs
    # Run all tests
    # test_implementations_match()
    
    # print("\n✓ All SingleInputEmbedding tests passed")
    # print("  - Output matching verification")
    # print("  - Shape verification")
    # print("  - Multiple input tests")

@beartype
def test_multiple_input_embedding() -> None:
    """Test JAX/Equinox MultipleInputEmbedding against PyTorch version"""
    
    # Set random seeds
    torch.manual_seed(0)
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    in_channels: List[int] = [8, 8]
    # in_channels = [2, 2]
    out_channel: int = 8
    batch_size: int = 2
    
    # Create models
    jax_model = JaxMultiEmbedding(in_channels, out_channel, key=key)
    torch_model = TorchMultipleInputEmbedding(in_channels=in_channels, out_channel=out_channel)
    
    # Create identical input data
    # continuous_inputs_jax: List[List[Float[Array, "_"]]] = []
    continuous_inputs_torch : List[Float[Array, "batch _"]] = []
    continuous_inputs_jax : List[Float[Array, "batch _"]] = []

    for in_channel in in_channels:
        np_input = np.random.randn(batch_size, in_channel).astype(np.float32)
        continuous_inputs_torch.append(torch.FloatTensor(np_input))
        continuous_inputs_jax.append(jnp.array(np_input))


    # We can use the torch input an vmap to axis 1
    # Then convert list to jax array

    # for _ in range(batch_size):
    #     batch_input = []
    #     # Create continuous inputs
    #     for in_channel in in_channels:
    #         np_input = np.random.randn(in_channel).astype(np.float32)
    #         batch_input.append(jnp.array(np_input))    
    #         # continuous_inputs_torch.append(torch.FloatTensor(np_input))
    #     continuous_inputs_jax.append(batch_input)


    # # Create batch inputs
    # for i, input in enumerate(in_channels):
    #     torch_list = []
    #     for inputs in continuous_inputs_jax:
    #         torch_list.append(inputs[i])
    #     # Do stack
    #     torch_list = torch.stack(torch_list)
    #     continuous_inputs_torch.append(torch_list)

    def jax_model_continuous(*continuous_inputs):
        return jax_model(continuous_inputs)


    # Get outputs
    jax_output = jax.vmap(jax_model_continuous)(*continuous_inputs_jax)  # Use vmap instead

    # jax_output = jax_model(continuous_inputs_jax)
    # print("Continuous inputs shapes:", [x.shape for x in continuous_inputs_torch])
    torch_output = torch_model(continuous_inputs_torch)

    @jax.jit
    def dummy_loss_multi(model, x):
        return jax.vmap(model)(x).sum()
    
    # Get gradients
    grads = jax.grad(dummy_loss_multi)(jax_model, continuous_inputs_jax)
    # print(f"Multiple Input Gradients:\n{grads}")
    
    # Convert outputs to numpy for comparison
    jax_np = np.array(jax_output)
    # torch_np = torch_output.detach().numpy()
    
    # Basic shape test
    assert jax_output.shape == (batch_size, out_channel)
    print("✓ Multiple Input Shape test passed")
    # print("Torch shape: ", torch_output.shape)
    # print("Jax shape: ", jax_output.shape)
    # ensure categorical is same columsn as output type
    

    # Fix categorical input test
    categorical_dim: int = out_channel
    cat_input_jax: Float[Array, "batch n cat_dim"] = jnp.array(
        np.random.randn(batch_size, 3, categorical_dim).astype(np.float32)
    )

    def jax_model_categorical(categorical_inputs, *continuous_inputs):
        return jax_model(continuous_inputs, categorical_inputs)


    jax_output_with_cat = jax.vmap(jax_model_categorical)(cat_input_jax, *continuous_inputs_jax)  # Use vmap instead # Mapping over the batches

    # print(f"PyTorch: {cat_input_torch.shape}")
    
    # Ensure continuous inputs haven't been modified
    # continuous_inputs_torch = [x.clone() for x in continuous_inputs_torch]  # Create fresh copies
    # Print shapes of continuous inputs
    # print("Continuous inputs shapes:", [x.shape for x in continuous_inputs_torch])
    # Continuous inputs torch needs to be torch.size([2,4])
    # print(f"Categorical input shape: {cat_input_torch.shape}")

    # torch_output_with_cat = torch_model(continuous_inputs_torch, cat_input_torch)
    
    assert jax_output_with_cat.shape == (batch_size, out_channel)
    print("✓ Multiple Input with Categorical Shape test passed")

if __name__ == "__main__":
    print("\nRunning Regular Tests:")
    test_single_input_embedding()
    # test_multiple_input_embedding()