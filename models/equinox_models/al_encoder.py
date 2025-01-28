import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding

from einops import rearrange, repeat, reduce

# Import beartype
from beartype import beartype
from typing import List, Tuple, Optional

from utils import print_array_type

import copy

from models.equinox_models.mlp import MLP, ReLU



class ALEncoder(eqx.Module):

    _lane_embed: eqx.nn.Embedding
    _is_intersection_embed: eqx.nn.Embedding
    _turn_direction_embed: eqx.nn.Embedding
    _traffic_control_embed: eqx.nn.Embedding

    mlp: MLP
    attention: eqx.nn.MultiheadAttention
    lin_self: eqx.nn.Linear
    attn_dropout: float

    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    embed_dim: int
    num_heads: int
    dropout: float



    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int, num_heads: int, dropout: float, *, key: PRNGKeyArray):

        keys = jax.random.split(key, 10)


        self._lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim, key=keys[0])
        self._is_intersection_embed = jax.random.normal(key, (2, embed_dim)) * 0.02  # matches PyTorch's std=.02    
        self._turn_direction_embed = jax.random.normal(keys[1], (3, embed_dim)) * 0.02  # matches PyTorch's std=.02
        self._traffic_control_embed = jax.random.normal(keys[2], (2, embed_dim)) * 0.02  # matches PyTorch's std=.02

        self.mlp = MLP(embed_dim, dropout, keys[3:5])
        self.attention = eqx.nn.MultiheadAttention(num_heads=num_heads, query_size=embed_dim, key=keys[6])
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[7])
        self.attn_dropout = dropout

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[8])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[9])

        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout


    def __call__(self, 
                 temporal_embeddings: Array, # [N, embed_dim]
                 positions: Float[Array, "N t=2 xy=2"],
                 is_intersection: Array, # [L]
                 turn_direction: Array, # [L]
                 traffic_control: Array, # [L]
                 *, 
                 key: PRNGKeyArray) -> Array:



        keys = jax.random.split(key, temporal_embeddings.shape[0])

        def f(pos, temporal_embedding, key) -> Array:

            return self.hub_spoke_nn(pos, temporal_embedding, is_intersection, turn_direction, traffic_control, key=key)
        
        outputs = jax.vmap(f)(
            positions,
            temporal_embeddings,
            keys
        )

        return outputs





    def hub_spoke_nn(
            self,
            temporal_embedding: Float[Array, "xy=2"],
            pos: Float[Array, "t=2 xy=2"],
            lane_start: Float[Array, "L xy=2"],
            lane_end: Float[Array, "L xy=2"],
            is_intersection: Array,
            turn_direction: Array,
            traffic_control: Array,
            *,
            key: PRNGKeyArray
    ) -> Array:

        jax.debug.breakpoint()

        # Compute the relative position

        # Create the neighbor mask

        rotate_mat = self.create_rotate_mat(pos[-1]-pos[0])


        # 1. create the neighbor mask

        agent_lane_rel_pos = lane_start - pos[-1]

        lane_vecs = lane_end - lane_start



        # 2. create the neighbor mask

        neighbor_mask = self.create_neighbor_mask(agent_lane_rel_pos)

        # 3. create the rotate mat
        
        # Fuse all features for the lane
        # Apply rotation to the lane vectors agent_lane_rel_pos, and lane_vecs
        # Do with vmap
        
        # Then concatenate all the features together
        #         nbr_embed = jax.vmap(lambda a, b: self._nbr_embed([a, b]))(
        #     neighbors_xy, neighbors_dxy
        # ) Do this but over all the features. It should be L indexed 5 features a b c d e 



        lane_features = self.lane_embed(lane_pos, rel_pos,  rotate_mat, is_intersection, turn_direction, traffic_control)



        



