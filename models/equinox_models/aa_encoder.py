import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding

from einops import rearrange, reduce

# Import beartype
from beartype import beartype
import typing
from typing import List, Tuple, Optional, Union

import numpy as np
from utils import print_array_type
import random

# Add jax type signature to inputs and outputs


class ReLU(eqx.Module):
    def __call__(self, x: Float[Array, "batch 8"], key=None):
        output = jax.nn.relu(x)  # Float[Array, "batch 8"]
        return output


class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    relu: ReLU

    # @beartype
    def __init__(self, embed_dim: int, dropout_rate: float, keys: PRNGKeyArray):
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[0])
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[1])
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)
        self.relu = ReLU()

    # @beartype
    def __call__(
        self, nodes, key  # Shape: [batch_size, node_dim=2] -> Float[Array, "2 2"]
    ) -> Float[Array, "2 2"]:  # Shape: [batch_size, node_dim=2] -> Float[Array, "2 2"]

        key1, key2 = jax.random.split(key)
        nodes = self.linear1(nodes)  # Float[Array, "batch 8"]

        nodes = self.relu(nodes)  # Float[Array, "batch 8"]

        nodes = self.dropout1(nodes, key=key1)  # Float[Array, "batch 8"]

        nodes = self.linear2(nodes)  # Float[Array, "batch 2"]
        nodes = self.dropout2(nodes, key=key2)  # Float[Array, "batch node_dim=2"]
        return nodes


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
    max_radius: int

    @beartype
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
        bos_key = jax.random.fold_in(keys[11], 0)  # PRNGKeyArray[Array, "2"]
        self.bos_token = (
            jax.random.normal(bos_key, shape=(self.historical_steps, self.embed_dim))
            * 0.02
        )  # Scale factor similar to PyTorch's default initialization Float[Array, "20 2"]

        # TODO: Add initialization for the weights
        # self.apply(init_weights)

    # @beartype
    def __call__(
        self,
        positions: Float[
            Array, "N 50 2"
        ],  # Full trajectories [Num_nodes, timesteps, xy]
        bos_mask: Bool[Array, "N 20"],  # Shape: [Numnodes, timesteps]
        t: Optional[int] = None,  # Optional[int]
    ):

        node_indices = jnp.arange(positions.shape[0])
        # outputs = jax.vmap(
        #     lambda idx: self.hub_spoke_nn(
        #         idx=idx,
        #         positions=positions,
        #         t=t,
        #         padding_mask=bos_mask,
        #     )
        # )(node_indices) # Calls function for each node
        # Start with regular loop
        outputs = []
        for idx in node_indices:
            outputs.append(self.hub_spoke_nn(idx, positions, t, bos_mask))
        return outputs

    def hub_spoke_nn(
        self,
        idx: int,
        positions: Float[Array, "N t=50 2"],
        t: int,
        padding_mask: Bool[Array, "node 50"],
    ) -> Float[Array, "hidden_dim"]:

        if t < 1:
            return jnp.zeros(positions.shape[0])

        dpositions = positions[:, t, :] - positions[:, t - 1, :]

        adj_mat = self.create_adj_matrix(
            positions[:, t, :], padding_mask[:, t]
        )  # Set all to true bool

        # 3. Get hub and spoke data
        node_xy = positions[idx, t, :]  # hub position
        node_dxy = dpositions[idx, :]  # hub vel

        rot_mat = jnp.eye(2)
        neighbors_xy = positions[adj_mat[idx]]  # <--- spokes
        neighbors_dxy = dpositions[adj_mat[idx]]

        neighbors_xy = jax.vmap(lambda xy: xy - node_xy)(neighbors_xy)
        del node_xy  # Don't need node_xy any more. It should be (0,0)
        node_dxy = node_dxy @ rot_mat

        neighbors_xy = jax.vmap(lambda n: n @ rot_mat)(neighbors_xy)

        center_embed = self._center_embed(node_dxy)

        nbr_embed = jax.vmap(lambda a, b: self._nbr_embed([a, b]))(
            neighbors_xy[:, t, :], neighbors_dxy
        )
        print("nbr_embed", nbr_embed.shape)
        print("nbr_embed", nbr_embed)
        breakpoint()

    def create_adj_matrix(
        self,
        positions: Float[Array, "N 2"],  # positions
        padding_mask: Bool[Array, "N"],  # mask at current timestep
    ) -> Bool[Array, "node node"]:
        """Creates adjacency matrix for nodes within max_radius and not padded."""
        # 1. Compute pairwise distances between all nodes
        # Expand dimensions for broadcasting
        pos_i = positions[:, None, :]  # [N, 1, 2]
        pos_j = positions[None, :, :]  # [1, N, 2]

        # Their difference gives [node, node, 2] representing vectors between all pairs
        diff = pos_i - pos_j  # [N, N, 2]

        # 2. Get actual distances between nodes
        distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # [node, node]

        # 3. Create three mask conditions:
        # a. Distance threshold
        dist_mask = distances <= self.max_radius

        # b. Both nodes must be valid (not padded)
        valid_mask = ~padding_mask[:, None] & ~padding_mask[None, :]

        # c. No self-connections
        ### Set alll top true
        no_self = ~jnp.eye(positions.shape[0], dtype=bool)

        # 4. Combine all conditions
        adj_mat = ~(dist_mask & valid_mask & no_self).astype(bool)

        return adj_mat

    def compute_rotation_matrices(self, positions, padding_mask, adj_mat):
        pass


# TODO write a function that creates an adjaceny matrix for time t it will then determine if padding mask is true or false. A node wont be connected if it is padded. Filter out things that are padding and things that are too far. This will tell us what are the hubs and spokes.
# Ignore x and just use the positions.
