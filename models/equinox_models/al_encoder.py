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
    _edge_embed: eqx.nn.Embedding
    _is_intersection_embed: eqx.nn.Embedding
    _turn_direction_embed: eqx.nn.Embedding
    _traffic_control_embed: eqx.nn.Embedding

    mlp: MLP
    attention: eqx.nn.MultiHeadAttention
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

        keys = jax.random.split(key, 9)


        self._lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channels=embed_dim, key=keys[0])
        self._is_intersection_embed = jax.random.normal(key, (2, embed_dim)) * 0.02  # matches PyTorch's std=.02    
        self._turn_direction_embed = jax.random.normal(keys[1], (3, embed_dim)) * 0.02  # matches PyTorch's std=.02
        self._traffic_control_embed = jax.random.normal(keys[2], (2, embed_dim)) * 0.02  # matches PyTorch's std=.02

        self.mlp = MLP(in_channels=embed_dim, out_channels=embed_dim, hidden_channels=[embed_dim, embed_dim], key=keys[3])
        self.attention = eqx.nn.MultiHeadAttention(embed_dim, num_heads, dropout, key=keys[4])
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])
        self.attn_dropout = dropout

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[6])

        self.norm1 = eqx.nn.LayerNorm(embed_dim, key=keys[7])
        self.norm2 = eqx.nn.LayerNorm(embed_dim, key=keys[8])

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout


    def __call__(self, 
                 x: Tuple[Array, Array],
                 is_intersection: Array, 
                 turn_direction: Array, 
                 traffic_control: Array,
                 *, 
                 key: PRNGKeyArray) -> Array:


        x_lane, x_actor = x

        actor_node_indices = jnp.arange(x_actor.shape[0])

        keys = jax.random.split(key, x_actor.shape[0])

        def f(actor_node_idx, key) -> Array:

            return self.hub_spoke_nn(actor_node_idx, x_lane, is_intersection, turn_direction, traffic_control, key=key)
        
        outputs = jax.vmap(f)(
            actor_node_indices,
            keys
        )

        return outputs
        


    def hub_spoke_nn(
            self,
            actor_node_idx: Array,
            x_lane: Array,
            is_intersection: Array,
            turn_direction: Array,
            traffic_control: Array,
            *,
            key: PRNGKeyArray
    ) -> Array:

        x_lane, x_actor = x

        
