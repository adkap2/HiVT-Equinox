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
    max_radius: float


    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int, num_heads: int, dropout: float, max_radius: float, *, key: PRNGKeyArray):

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
        self.max_radius = max_radius

    def __call__(self, 
                 temporal_embeddings: Array, # [N, embed_dim]
                 positions: Float[Array, "N t=2 xy=2"],
                 lane_start: Float[Array, "L xy=2"],
                 lane_end: Float[Array, "L xy=2"],
                 is_intersection: Array, # [L]
                 turn_direction: Array, # [L]
                 traffic_control: Array, # [L]
                 *, 
                 key: PRNGKeyArray) -> Array:

        keys = jax.random.split(key, positions.shape[0])


        def f(pos, key) -> Array:
            return self.hub_spoke_nn(temporal_embeddings=temporal_embeddings, pos=pos, lane_start=lane_start, lane_end=lane_end, is_intersection=is_intersection, turn_direction=turn_direction, traffic_control=traffic_control, key=key)
        
        outputs = jax.vmap(f)(
            positions,
            keys
        )

        return outputs





    def hub_spoke_nn(
            self,
            temporal_embeddings: Float[Array, "xy=2"],
            pos: Float[Array, "t=2 xy=2"],
            lane_start: Float[Array, "L xy=2"],
            lane_end: Float[Array, "L xy=2"],
            is_intersection: Array,
            turn_direction: Array,
            traffic_control: Array,
            *,
            key: PRNGKeyArray
    ) -> Array:

        # jax.debug.breakpoint()

        # Compute the relative position

        # Create the neighbor mask

        rotate_mat = self.create_rotate_mat(pos[-1]-pos[0])


        # 1. create the neighbor mask

        agent_lane_rel_pos = lane_start - pos[-1]

        lane_vecs = lane_end - lane_start

        # 2. create the neighbor mask

        neighbor_mask = self.create_neighbor_mask(agent_lane_rel_pos)
        # jax.debug.breakpoint()
        # 3. create the rotate mat
        
        # Fuse all features for the lane
        # Apply rotation to the lane vectors agent_lane_rel_pos, and lane_vecs

        # So lan_vecs is a L x 2 array and rotate_mat is a 2 x 2 array
        rotated_lane_vecs = jnp.einsum('ij,lj->li', rotate_mat, lane_vecs)

        # agent_lane_rel_pos is a L x 2 array and rotate_mat is a 2 x 2 array
        rotated_agent_lane_rel_pos = jnp.einsum('ij,lj->li', rotate_mat, agent_lane_rel_pos)


        # DO i need to expand out is_intersection, turn_direction, traffic_control? to have the same shape as rotated_lane_vecs?

        # For [L, 2] shape:
        is_intersection = repeat(is_intersection, 'L -> L 2')
        turn_direction = repeat(turn_direction, 'L -> L 2')
        traffic_control = repeat(traffic_control, 'L -> L 2')

        # jax.debug.breakpoint()
        lane_features = jax.vmap(lambda *features: self._lane_embed(list(features)))(
            rotated_lane_vecs,
            rotated_agent_lane_rel_pos,
            is_intersection,
            turn_direction,
            traffic_control
        )

        # Apply the neighbor mask
        lane_features = jnp.where(neighbor_mask[:, None], lane_features, 0.0)

        # Apply attention
        query = rearrange(temporal_embeddings, "d -> 1 d")

        neighbor_mask = rearrange(neighbor_mask, 'L -> 1 L')
        # jax.debug.breakpoint()
        mha = self.attention(query=query, key_=lane_features, value=lane_features, mask=neighbor_mask)

        new_msg = rearrange(mha, "1 d -> d")


        gate = jax.nn.sigmoid(
            self.lin_ih(new_msg) + self.lin_hh(temporal_embeddings)
        )

        output = new_msg + gate * (
            self.lin_self(temporal_embeddings) - new_msg
        )

        output = output + self.mlp(self.norm2(output), key=key)

        return output

        # lane_features = self.lane_embed(lane_pos, rel_pos,  rotate_mat, is_intersection, turn_direction, traffic_control)



        


    def create_rotate_mat(self, dpos: Float[Array, "xy=2"]) -> Float[Array, "2 2"]:
        """Create rotation matrix from displacement vector.
        
        Args:
            dpos: Displacement vector [2] representing direction of motion
            
        Returns:
            rotate_mat: 2x2 rotation matrix that aligns with direction of motion
        """

        # jax.debug.breakpoint()
        # Compute rotation angle from displacement vector
        rotate_angle = jnp.arctan2(dpos[1], dpos[0])  # scalar
        
        # Compute sin and cos values
        sin_val = jnp.sin(rotate_angle)  # scalar
        cos_val = jnp.cos(rotate_angle)  # scalar

        # Create rotation matrix using jnp.array instead of zeros + at
        rotate_mat = jnp.array([
            [cos_val, -sin_val],
            [sin_val,  cos_val]
        ])
        
        return rotate_mat
    

    def create_neighbor_mask(self, agent_lane_rel_pos: Float[Array, "L xy=2"]) -> Bool[Array, "L xy=2"]:
        """Create neighbor mask based on agent's lane relative position.
        
        Args:
            agent_lane_rel_pos: Relative position of agent to lane [2]
            
        Returns:
            neighbor_mask: Boolean mask indicating if agent is in the lane
        """
        # return jnp.linalg.norm(agent_lane_rel_pos) < self.max_radius  # Example threshold
        # Calculate norm along last axis (xy dimension) for each lane
        distances = jnp.linalg.norm(agent_lane_rel_pos, axis=1)  # [L]
        
        # Create mask where True means "keep this lane"
        neighbor_mask = distances < self.max_radius  # [L]
        
        # Optional: use rearrange to make shape explicit
        neighbor_mask = rearrange(neighbor_mask, 'L -> L')
        
        return neighbor_mask
