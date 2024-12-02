import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding

from einops import rearrange, reduce
# Import beartype
from beartype import beartype

# Add jax type signature to inputs and outputs

class ReLU(eqx.Module):
    def __call__(self, x, key=None):
        return jax.nn.relu(x)


class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    relu: ReLU

    @beartype
    def __init__(self, embed_dim, dropout_rate, keys):
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[0])
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[1])
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)
        self.relu = ReLU()

    @beartype
    def __call__(self, x, key):
        key1, key2 = jax.random.split(key)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x, key=key1)
        x = self.linear2(x)
        x = self.dropout2(x, key=key2)
        return x


class AAEncoder(eqx.Module):
    _center_embed: SingleInputEmbedding
    _nbr_embed: MultipleInputEmbedding
    lin_q: eqx.nn.Linear
    lin_k: eqx.nn.Linear
    lin_v: eqx.nn.Linear
    lin_self: eqx.nn.Linear
    attn_dropout: float
    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    proj_drop: float
    # Layer Norms
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    # Historical steps
    historical_steps: int
    embed_dim: int
    num_heads: int
    dropout: float
    mlp: eqx.nn.Sequential
    bos_token: jnp.ndarray

    @beartype
    def __init__(
        self,
        historical_steps,
        node_dim, #TODO Node dim is always 2
        edge_dim,
        embed_dim,
        num_heads=8,
        dropout=0.1,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):

        keys = jax.random.split(key, 12)

        # print(f"keys shape: {keys.shape}")
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self._center_embed = SingleInputEmbedding(node_dim, embed_dim, key=keys[0])
        self._nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim, key=keys[1]
        )

        self.lin_q = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.lin_k = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        self.lin_v = eqx.nn.Linear(embed_dim, embed_dim, key=keys[4])
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])
        self.attn_dropout = eqx.nn.Dropout(dropout)

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[6])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[7])
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[8])
        self.proj_drop = eqx.nn.Dropout(dropout)

        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, dropout, keys[9:11])

        # Key differences from PyTorch:
        # Uses jnp.ndarray instead of torch.Tensor
        # Initialization is explicit using JAX's random number generator
        # No need for nn.Parameter as Equinox automatically treats array attributes as parameters
        # Initialize BOS token with random values
        bos_key = jax.random.fold_in(keys[11], 0)
        self.bos_token = (
            jax.random.normal(bos_key, shape=(self.historical_steps, self.embed_dim))
            * 0.02
        )  # Scale factor similar to PyTorch's default initialization

        # TODO: Add initialization for the weights
        # self.apply(init_weights)

    @beartype
    def __call__(
        self,
        x, # Shape: [batch_size, node_dim]
        edge_index, # shape: [2, num_edges]
        edge_attr, # Shape: [num_edges, edge_dim]
        bos_mask, # Shape: [batch_size]
        t: Optional[int] = None,
        rotate_mat=None, #shape: [batch_size, embed_dim, embed_dim] #TODO Every rotation matrix is a 2x2 matrix
        size=None,
    ):

        # CHeange type signature to [List] node features, edge features,
        # Break the call function into three steps
        # 1. Center Embedding
        # 2. Neighbor Embedding
        # 3. Multi-Head Attention propagate messages

        # Apply message function on all neighbors (just a vmap)

        # After when you do the aggregation, just a sum over all the messages where you take an inner product with the row and the vector messages
        # Altertively you can use where. If its 1 or 0 and when if it's 0 then you just 0 it.
        # Keep it in form [node, node, channel]
        # After applyig sum it will collopse to [node, channel] # Einops sum

        # 4. Feedforward
        # 5. Residual Connection
        # 6. Layer Norm

        #TODO Rotation matrix is always a 2x2 matrix
        # TODO this should be a tensor containing 2x2 matricies 
        # TODO edges themselves are just 2x2 since they are jsut distances
        # Want to apply rotation matrix to every node. 
        # Vmapping over the matrix multiplication operator

        # print("EQX X", x)

        if rotate_mat is None:
            # print("x.shape", x.shape)
            center_embed = self._center_embed(x)
        else:
            # TODO why do we have to expand dims to match rotation matrix
            # Look at what the dimensions represent

            # Do the vmap right here instead of rotation matrix:
            x_rotated = jax.vmap(lambda x, m: x @ m)(x, rotate_mat)
            # Vmap over the rotation matrix
            center_embed = jax.vmap(self._center_embed)(x_rotated)

        # Apply bos mask
        # breakpoint()
        # Apply bos mask using einops
        # Fix shapes for broadcasting
        bos_mask = rearrange(bos_mask, "n -> n 1")

        center_embed = jnp.where(bos_mask, self.bos_token[t], center_embed)

        center_embed = center_embed + self.create_message(
            jax.vmap(self.norm1)(center_embed), x, edge_index, edge_attr, rotate_mat
        )

        center_embed = jax.vmap(self.norm2)(center_embed)

        # TODO Talk to Marcell about this approach for handling keys with batch size
        # Testing
        batch_size = center_embed.shape[0]
        mlp_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

        center_embed = center_embed + jax.vmap(lambda x, k: self.mlp(x, k))(
            center_embed, mlp_keys
        )

        return center_embed

    @beartype
    def create_message(self,
                        center_embed, # Shape: [batch_size, embed_dim]
                        x, # Shape: [batch_size, node_dim]
                        edge_index, # Shape: [2, num_edges]
                        edge_attr, # Shape: [num_edges, edge_dim]
                        rotate_mat): # Shape: [batch_size, embed_dim, embed_dim]

        
        # print("EQX center_embed", center_embed)
        # Rotation matrix is a [2,2] tensor
        # All 2,2 tensors
        # TODO switch to rel pos 2 neightbor
        # print("EQX x.shape", x.shape)
        # print("EQX x", x)
        # Create a message funiton
        # First rotate the relative position by the rotation matrix
        # print("EQX x", x)
        x_j = x[edge_index[0]]  # Get source node features
        # print("EQX x_j.shape", x_j.shape)
        # print("EQX x[j]", x_j)
        # Get center rotate mat
        center_rotate_mat = rotate_mat[edge_index[1]]
        # print("EQX center_rotate_mat", center_rotate_mat)

        # Rotate node features
        x_rotated = rearrange(x_j, "n f -> n 1 f") @ center_rotate_mat
        x_rotated = rearrange(x_rotated, "n 1 f -> n f")

        # Rotate edge features
        edge_rotated = rearrange(edge_attr, "n f -> n 1 f") @ center_rotate_mat
        edge_rotated = rearrange(edge_rotated, "n 1 f -> n f")

        # print("EQX edge_rotated", edge_rotated)
        # print("EQX X_rotated", x_rotated)
        # Compute neighbor embedding
        # nbr_embed = self._nbr_embed([x_rotated, edge_rotated])
        # Do vmap here
        # nbr_embed = jax.vmap(self._nbr_embed)([x_rotated, edge_rotated])
        # x_rotated: [batch_size, node_dim]

        nbr_embed = jax.vmap(lambda x, e: self._nbr_embed([x, e]))(x_rotated, edge_rotated)

        # print("EQX nbr_embed", nbr_embed)

        # Questionable output shape for nbr_embed
        # Ensure identical initialization of weights and inputs
        # Check the LayerNorm and activation functions in the embedding networks
        # Verify the rotation matrix application is identical

        # print("EQX center_embed", center_embed)
        # Jax random key is different than torch so can expect slightly different results on the normalization
        # query = rearrange(
        #     self.lin_q(center_embed), "n (h d) -> n h d", h=self.num_heads
        # )
        query = rearrange(jax.vmap(lambda x: self.lin_q(x))(center_embed), "n (h d) -> n h d", h=self.num_heads)
        # print("EQX linq", jax.vmap(self.lin_q)(center_embed))
        # print("EQX query", query)
        # print("EQX query", query)
        # Jax random key is different than torch so can expect slightly different results on the normalization
        
        # old way
        # key = rearrange(self.lin_k(nbr_embed), "n (h d) -> n h d", h=self.num_heads)
        
        key = rearrange(jax.vmap(lambda x: self.lin_k(x))(nbr_embed), "n (h d) -> n h d", h=self.num_heads)
        # print("EQX key", key)
        # TODO check if this is correct with Marcell
        # Jax random key is different than torch so can expect slightly different results on the normalization
        value = rearrange(jax.vmap(lambda x: self.lin_v(x))(nbr_embed), "n (h d) -> n h d", h=self.num_heads)
        # print("EQX value", value)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        # print("EQX scale", scale)

        alpha = (query * key).sum(axis=-1) / scale
        print("EQX alpha", alpha)
        # Do softmax
        alpha = jax.nn.softmax(alpha)

        # TODO Pass a key into this function as an additional random key
        alpha = self.attn_dropout(alpha, key=jax.random.PRNGKey(0))

        messages = value * rearrange(alpha, "n h -> n h 1")

        # aggregate
        messages = reduce(messages, "n h d -> n d", "sum")  # This reduces to [b, d]
        ## PROPAGATION IS DONE
        # print("EQX messages", messages)

        # Do equivilant of self.out_proj
        # Which is linear transformation to aggregated messaged
        messages = self.out_proj(messages)
        # print("EQX messages", messages)

        # print("EQX messages", messages)

        # Apply dropout
        # TODO pass a key into this function as an additional random key
        messages = self.proj_drop(messages, key=jax.random.PRNGKey(0))

        # THen update logic to add to center_embed
        gate = jax.nn.sigmoid(self.lin_ih(messages) + self.lin_hh(center_embed))

        messages = messages + gate * (self.lin_self(center_embed) - messages)
        # print("EQX messages", messages)

        return messages

        # Everything goes into message function
        # center_rotate_mat = rotate_mat[edge_index[1]]
