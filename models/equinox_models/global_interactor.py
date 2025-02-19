import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool, Scalar
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding
from models.equinox_models.mlp import MLP
from utils import TemporalData

from einops import rearrange, repeat, reduce

# Import beartype
from beartype import beartype
from typing import List, Tuple, Optional

from utils import print_array_type

import copy

# Add jax type signature to inputs and outputs
# TODO use bert classifier but dont return classifier head and dont tokenize

# Looking at BERT encoder, we take the sequence and output the dictionary array except we don't want that output
# We want 

#TODO use same building blocks and do global connections as transformer 
# Global connections are standard transofrmer, then just do a softmax between the agents
# Is just a Multihead attention
# Instead of neighbors do everybody
# Similar to AA
# DO UNIT TESTS FOR ALL INPUT FEATURES 
# Making sure im feeding the right things into each layer

# TODO do the same thing for the local connections

# We want hub spoke
# Agent i is the hub, and all other agents are the spokes

# Theres no radius, no neighbors, just everyone

# Relative positions 

class GlobalInteractorLayer(eqx.Module):
    pass


class GlobalInteractor(eqx.Module):

    historical_steps: int
    embed_dim: int
    num_modes: int

    num_heads: int
    num_layers: int
    dropout: float
    edge_dim: int
    
    multihead_proj: eqx.nn.Linear

    rel_embed: MultipleInputEmbedding
    global_interactor_layers: List[GlobalInteractorLayer]
    norm: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, 
                 historical_steps: int, 
                 embed_dim: int, 
                 edge_dim: int, 
                 num_modes: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float, 
                 *, 
                 key: PRNGKeyArray):
        super(GlobalInteractor, self).__init__()

        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.edge_dim = edge_dim

        keys = jax.random.split(key, 4)

        self.multihead_proj = eqx.nn.Linear(embed_dim, num_modes * embed_dim, key=keys[0])
        # Can i skip MultipleInputEmbedding since we concernced with global attention between all agents attending to one another
        self.rel_embed = MultipleInputEmbedding(in_channels=[edge_dim, edge_dim], out_channel=embed_dim, key=keys[0])

        jax_layer_keys = jax.random.split(keys[1], num_layers)
        self.global_interactor_layers = [GlobalInteractorLayer(num_modes=num_modes, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, key=jax_layer_keys[i]) for i in range(num_layers)]
        self.norm = eqx.nn.LayerNorm(embed_dim)

        self.norm2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self,
                data: TemporalData, 
                local_embed: Float[Array, "F N D"],
                *,
                key: PRNGKeyArray
                ) ->  Float[Array, "F N D"]:
        
        # We need the relative positiopons embeds 

        # I am going to vmap over the nodes, then the key and the value 
        # Need to compute the mlp edge feature similar to the AA encoder 
        # now have two need features which are the sin and co which are the change in angles
        # Should pre compute the sin and cos of the relative angles 
        # Actually dont worry about the sin yet 


        # Need the multi input embedding to take the temporal embedding and the relative position embedding

        # I need to rotate Pj and Pi into the agents frame and its axis aligned


        positions = data['positions']  # [N, T, 2]
        t = self.historical_steps - 1  # Use last historical step
        keys = jax.random.split(key, positions.shape[0])

        
        # Create node indices for vmapping
        node_indices = jnp.arange(positions.shape[0]) 

        # Vmap over all nodes as hubs
        def f(idx, key):
            return self.hub_spoke_attention(
                idx=idx,
                positions=positions,
                local_embed=local_embed,
                t=t,
                key=key
            )
        
        outputs = jax.vmap(f)(node_indices, keys)
        # TODO I need to handle this rearange better. I dont feel comfortable with the shapes here
        # This is a (9,2,9,2)
        # jax.debug.breakpoint()
        return outputs

    def hub_spoke_attention(self,
                            idx: Int[Array, ""],
                            positions: Float[Array, "N t=50 xy=2"],
                            local_embed: Float[Array, "F N D"],
                            t: Int[Scalar, ""],
                            key: PRNGKeyArray
                            ) -> Float[Array, "hidden_dim"]:
        
        
        keys = jax.random.split(key, 2)

        # 1. Get trajectory vectors for current and previous timestep
        curr_positions = positions[:, t, :]    # [N, 2]
        prev_positions = positions[:, t-1, :]  # [N, 2]
        dpositions = curr_positions - prev_positions  # [N, 2] movement vectors

        # 2. Calculate heading for agent i (hub)
        hub_vector = dpositions[idx]  # [2]
        theta_i = jnp.arctan2(hub_vector[1], hub_vector[0])

        # 3. Create rotation matrix for axis alignment
        cos_i, sin_i = jnp.cos(theta_i), jnp.sin(theta_i)
        R_i = jnp.array([
            [cos_i, -sin_i],
            [sin_i, cos_i]
        ])  # [2, 2]

        # 4. Calculate relative positions and align to hub's axis
        rel_positions = curr_positions - curr_positions[idx]  # [N, 2]
        rel_positions_aligned = jax.vmap(lambda p: R_i.T @ p)(rel_positions)  # [N, 2]

        # 5. Calculate relative headings and align
        thetas_j = jnp.arctan2(dpositions[:, 1], dpositions[:, 0])  # [N]
        delta_theta = thetas_j - theta_i  # [N]

        # 6. Create rotation embeddings (aligned to hub's axis)
        theta_embed = jnp.stack([
            jnp.cos(delta_theta),
            jnp.sin(delta_theta)
        ], axis=-1)  # [N, 2]

        # 7. Combine aligned features
        rel_embed = jax.vmap(lambda pos, angle: self.rel_embed([pos, angle]))(
            rel_positions_aligned,  # Axis-aligned positions
            theta_embed            # Axis-aligned angles
        )

        output = local_embed[:,idx,:]

        keys = jax.random.split(keys[1], len(self.global_interactor_layers))
        for layer, key in zip(self.global_interactor_layers, keys):
            output = layer(x=output, rel_embed=rel_embed, key=key)


        vmapped_norm2 = jax.vmap(self.norm2)

        output = vmapped_norm2(output)

        vmapped_multihead_proj = jax.vmap(self.multihead_proj)
        
        output = vmapped_multihead_proj(output)
        # Explode the output to the right shape
        
        # TODO I need to handle this rearange better. I dont feel comfortable with the shapes here
        output = rearrange(output, 'n (f d) -> f n d', f=self.num_modes)
        # THis is a (2,9,2)

        # jax.debug.breakpoint()


        return output


    def compute_rotation_matrix(self, velocity: Float[Array, "2"]) -> Float[Array, "2 2"]:
        angle = jnp.arctan2(velocity[1], velocity[0])
        cos_a = jnp.cos(angle)
        sin_a = jnp.sin(angle)
        rotate_mat = jnp.zeros((2, 2))
        rotate_mat = rotate_mat.at[0, 0].set(cos_a)
        rotate_mat = rotate_mat.at[0, 1].set(-sin_a)
        rotate_mat = rotate_mat.at[1, 0].set(sin_a)
        rotate_mat = rotate_mat.at[1, 1].set(cos_a)
        return rotate_mat



class GlobalInteractorLayer(eqx.Module):

    num_modes: int
    embed_dim: int
    num_heads: int
    dropout: float

    # Attention and linear layers
    self_attn: eqx.nn.MultiheadAttention
    lin_self: eqx.nn.Linear
    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear
    
    # Normalization and dropout
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    
    # MLP layers
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self,
                 num_modes: int,
                 embed_dim: int, 
                 num_heads: int, 
                 dropout: float, 
                 *, 
                 key: PRNGKeyArray):
        

        keys = jax.random.split(key, num_heads)

        self.num_modes = num_modes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout


        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=self.num_heads,
            query_size=self.embed_dim, 
            key=keys[0]
        )
        
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        
        # Normalization and dropout
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)
        self.dropout1 = eqx.nn.Dropout(dropout)
        self.dropout2 = eqx.nn.Dropout(dropout)

        # MLP layers
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[4])
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[5])



    def __call__(self,
                 x: Float[Array, "N D"], 
                 rel_embed: Float[Array, "N D"], 
                 *, 
                 key: PRNGKeyArray) -> Float[Array, "N D"]:

        keys = jax.random.split(key, 3)


        vmapped_norm1 = jax.vmap(self.norm1)
        vmapped_norm2 = jax.vmap(self.norm2)

        vmapped_lin_ih = jax.vmap(self.lin_ih)
        vmapped_lin_hh = jax.vmap(self.lin_hh)
        vmapped_lin_self = jax.vmap(self.lin_self)

        x_norm = vmapped_norm1(x)

        attn_out = self.self_attn(
            query=x_norm,   # [N, D]
            key_=rel_embed, # [N, D]
            value=rel_embed, # [N, D]
            key=keys[0]
        )


        # GRU-like update
        gate = jax.nn.sigmoid(
            vmapped_lin_ih(attn_out) + 
            vmapped_lin_hh(x)
        )
        
        x = attn_out + gate * (vmapped_lin_self(x) - attn_out)

        x = x + self._ff_block(vmapped_norm2(x), key=keys[1])


        return x

    def _ff_block(self,
        x: Float[Array, "N D"], 
        *, 
        key: PRNGKeyArray) -> Float[Array, "N D"]:
        
        key1, key2 = jax.random.split(key)
        

        vmapped_linear1 = jax.vmap(self.linear1)
        vmapped_linear_2 = jax.vmap(self.linear2)


        x = vmapped_linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout1(x, key=key1)
        x = vmapped_linear_2(x)
        x = self.dropout2(x, key=key2)
        
        return x

