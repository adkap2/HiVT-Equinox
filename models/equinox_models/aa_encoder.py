import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool, Union, Scalar
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding

from einops import rearrange, reduce

# Import jnp

# Import beartype
from beartype import beartype
from typing import List, Tuple, Optional

from utils import print_array_type

# Add jax type signature to inputs and outputs

from models.equinox_models.mlp import MLP, ReLU


@beartype
class AAEncoder(eqx.Module):
    _center_embed: SingleInputEmbedding
    _nbr_embed: MultipleInputEmbedding
    attention: eqx.nn.MultiheadAttention
    lin_self: eqx.nn.Linear
    attn_dropout: float
    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear
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
    max_radius: int

    def __init__(
        self,
        historical_steps: int,
        node_dim: int,  # TODO Node dim is always 2
        edge_dim: int,  # 2
        embed_dim: int,  # 2
        num_heads: int = 8,
        dropout: float = 0.1,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):

        keys = jax.random.split(key, 12)

        self.max_radius = 50
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self._center_embed = SingleInputEmbedding(node_dim, embed_dim, key=keys[0])
        self._nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim, key=keys[1]
        )

        self.attention = eqx.nn.MultiheadAttention(
            num_heads=self.num_heads, query_size=self.embed_dim, key=keys[2]
        )

        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        self.attn_dropout = eqx.nn.Dropout(dropout)

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[4])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])

        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

        self.mlp = MLP(embed_dim, dropout, keys[6:8])

        # Initialize BOS token with random values
        bos_key = jax.random.fold_in(keys[8], 0)  # PRNGKeyArray[Array, "2"]
        self.bos_token = (
            jax.random.normal(bos_key, shape=(self.historical_steps, self.embed_dim))
            * 0.02
        )  # Scale factor similar to PyTorch's default initialization Float[Array, "20 2"]

        # TODO: Add initialization for the weights
        # self.apply(init_weights)

    def __call__(
        self,
        positions: Float[
            Array, "N t=50 xy=2"
        ],  # Full trajectories [Num_nodes, timesteps, xy]
        bos_mask: Bool[Array, "N t=20"],  # Shape: [Numnodes, timesteps]
        padding_mask: Bool[Array, "N t=50"],  # Shape: [Numnodes, timesteps]
        t: Int[Scalar, ""],
        key: PRNGKeyArray,
    ) -> Float[Array, "N hidden_dim"]:

        # jax.debug.breakpoint()

        # T is a trivial array
        # jax.debug.print("t {t}", t=t)
        # assert (t > 0).all(), "t must be greater than 0"

        # TODO create a key split of position shape.siuze
        node_indices = jnp.arange(0, positions.shape[0])

        keys = jax.random.split(key, positions.shape[0])

        def f(idx, key):
            return self.hub_spoke_nn(idx, positions, t, bos_mask, padding_mask, key)

        outputs = jax.vmap(f)(node_indices, keys)

        return outputs

    # TODO make own classic nn

    # Unit test example count number of neighbors, then computer avg distance
    # Tell furthest neighbors

    def hub_spoke_nn(
        self,
        idx: Int[Array, ""],
        positions: Float[Array, "N t=50 xy=2"],
        t: Int[Scalar, ""],
        bos: Bool[Array, "N t=20"],
        padding_mask: Bool[Array, "N t=50"],
        key: PRNGKeyArray,
    ) -> Float[Array, "hidden_dim"]:  # TODO complete function before adding this in

        # Split it here the key
        # TODO pass a key into this function

        # assert (t > 0).all(), "t must be greater than 0"

        dpositions = positions[:, t, :] - positions[:, t - 1, :]

        rot_mat = self.compute_rotation_matrix(dpositions[idx])  # [2]

        mask = self.create_neighbor_mask(
            idx, positions[:, t, :], padding_mask[:, t], bos[:, t]
        )  # Set all to true bool

        # 3. Get hub and spoke data
        node_xy = positions[idx, t, :]  # hub position
        node_dxy = dpositions[idx, :]  # hub vel

        # Apply operations on dense matrix, then filter out during the attention step
        neighbors_xy = positions[:, t, :]  # <--- spokes
        neighbors_dxy = (
            dpositions  # If node has no neighbors, this may happen then handle that
        )

        neighbors_xy = jax.vmap(lambda xy: xy - node_xy)(neighbors_xy)
        del node_xy  # Don't need node_xy any more. It should be (0,0)
        # node_dxy = node_dxy @ rot_mat # Want to make sure we are rotating to nodexy coordinates
        node_dxy = rot_mat @ node_dxy

        # neighbors_xy = jax.vmap(lambda n: n @ rot_mat)(neighbors_xy)
        neighbors_xy = jax.vmap(lambda n: rot_mat @ n)(neighbors_xy)

        center_embed = self._center_embed(node_dxy)
        center_embed = self.norm1(center_embed)

        # Expand out by 1 element

        nbr_embed = jax.vmap(lambda *features: self._nbr_embed(list(features)))(
            neighbors_xy, neighbors_dxy
        )

        query = rearrange(center_embed, "d -> 1 d")

        # MHA
        mha = self.attention(query=query, key_=nbr_embed, value=nbr_embed, mask=mask)

        new_msg = rearrange(mha, "1 d -> d")
        # Rearange center embed
        gate = jax.nn.sigmoid(self.lin_ih(new_msg) + self.lin_hh(center_embed))

        # It is a convex combination
        center_embed = new_msg + gate * (
            self.lin_self(center_embed) - new_msg
        )  # Shape: [hidden_dim = 2]

        center_embed = self.norm2(center_embed)

        ff_output = self.mlp(center_embed, key=key)

        center_embed = center_embed + ff_output

        return center_embed

    def create_neighbor_mask(
        self,
        idx: Int[Array, ""],
        positions: Float[Array, "N 2"],
        padding_mask: Bool[Array, "N"],
        bos_mask: Bool[Array, "N"],
    ) -> Bool[Array, "1 N"]:
        """Creates adjacency matrix for nodes within max_radius and not padded."""

        # 1. Calculate relative positions
        rel_pos = positions[idx] - positions

        # 2. Calculate distances
        dist = jnp.linalg.norm(rel_pos, ord=2, axis=1)

        # 3. Create distance mask
        dist_mask = dist <= self.max_radius

        dist_mask = rearrange(dist_mask, "N -> 1 N")

        # 4. Create valid mask
        valid_mask = ~padding_mask

        valid_mask = rearrange(valid_mask, "N -> 1 N")

        # 5. Create self connections
        self_connections = jnp.eye(1, positions.shape[0], dtype=bool)

        # 6. Combine masks
        adj_mat = (dist_mask & valid_mask).astype(bool)

        adj_mat |= self_connections

        return adj_mat

    def compute_rotation_matrix(
        self,
        dpositions: Float[Array, "xy=2"],
    ) -> Float[Array, "2 2"]:

        # TODO CHECK with Marcell if this is correct
        # Get displacement vector for the specific node

        # Compute rotation angle from displacement vector
        rotate_angle = jnp.arctan2(dpositions[1], dpositions[0])  # scalar

        # Compute sin and cos values
        sin_val = jnp.sin(rotate_angle)  # scalar
        cos_val = jnp.cos(rotate_angle)  # scalar

        # Create rotation matrix for the single node
        rotate_mat = jnp.zeros((2, 2))
        rotate_mat = rotate_mat.at[0, 0].set(cos_val)
        rotate_mat = rotate_mat.at[0, 1].set(-sin_val)
        rotate_mat = rotate_mat.at[1, 0].set(sin_val)
        rotate_mat = rotate_mat.at[1, 1].set(cos_val)

        return rotate_mat


# TODO write a function that creates an adjaceny matrix for time t it will then determine if padding mask is true or false. A node wont be connected if it is padded. Filter out things that are padding and things that are too far. This will tell us what are the hubs and spokes.
# Ignore x and just use the positions.
