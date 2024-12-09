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


# Add jax type signature to inputs and outputs


class ReLU(eqx.Module):
    def __call__(self,
                  x: Float[Array, "batch 8"],
                    key=None):
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
        self, 
        nodes,  # Shape: [batch_size, node_dim=2] -> Float[Array, "2 2"]
        key
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

    # @beartype
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


    #TODO def hub_spoke_nn(self,
    # idx: int,
    # *,
    # rot_mats: Optional[Float[Array, "node 2 2"]] = None,
    #nodes: Float[Array, "node node_dim=2"],
    # adj_mat: Bool[Array, "node node"],


    # node = nodes[idx]
    # neighbors = nodes[adj_mat[idx]]
    # R = rots_mats[idx]

    # # 2. Translate to node centric frame
    # ## 2a center on node

    # ## 2b apply rotation matrices

    # node = node@R
    # neighbors = jax.vmap(lambda n: n @ R)(neighbors)

    # center_embed = self._center_embed(node)


     




    #  node: Float[Array, "node_dim=2"],
    #  adj_row: Bool[Array, "node"],
    #  nodes: Float[Array, "node node_dim=2"]) -> Float[Array, "hidden_dim"]

    @beartype
    def __call__(
        self,
        nodes: Float[Array, "batch_size node_dim=2"],  # Shape: [batch_size, node_dim=2]
        edge_index: Int[Array, "2 num_edges"],  # shape: [2, num_edges]
        edge_attr: Float[Array, "num_edges 2"],  # Shape: [num_edges, edge_dim]
        bos_mask: Bool[Array, "batch_size"],  # Shape: [batch_size]
        t: Optional[int] = None,  # Optional[int]
        rotate_mat: Optional[Float[Array, "batch_size 2 2"]] = None,  # shape: [batch_size, embed_dim, embed_dim]
        size=None,
    ):

        
        #TODO message_passing = jax.vmap(sel.hub_spoke_nn)
        #Todo latent = message_passing(nodes, adj_mat)

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

        # TODO Rotation matrix is always a 2x2 matrix
        # TODO this should be a tensor containing 2x2 matricies
        # TODO edges themselves are just 2x2 since they are jsut distances
        # Want to apply rotation matrix to every node.
        # Vmapping over the matrix multiplication operator

        # print("EQX X", x)

        if rotate_mat is None:
            # print("x.shape", x.shape)
            center_embed = self._center_embed(nodes)
        else:
            # TODO why do we have to expand dims to match rotation matrix
            # Look at what the dimensions represent

            # Do the vmap right here instead of rotation matrix:
            x_rotated = jax.vmap(lambda x, m: x @ m)(
                nodes, rotate_mat
            )  # Float[Array, "2 2"]

            # Vmap over the rotation matrix
            center_embed = jax.vmap(self._center_embed)(
                x_rotated
            )  # Float[Array, "2 2"]

        # Apply bos mask
        # breakpoint()
        # Apply bos mask using einops
        # Fix shapes for broadcasting
        bos_mask = rearrange(bos_mask, "n -> n 1")  # Float[Array, "2 1"]

        center_embed = jnp.where(
            bos_mask, self.bos_token[t], center_embed
        )  # Float[Array, "2 2"]

        center_embed = center_embed + self.create_message(
            jax.vmap(self.norm1)(center_embed), nodes, edge_index, edge_attr, rotate_mat
        )  # Float[Array, "2 2"]

        center_embed = jax.vmap(self.norm2)(center_embed)  # Float[Array, "2 2"]

        # TODO Talk to Marcell about this approach for handling keys with batch size
        # Testing
        batch_size = center_embed.shape[0]  # Int
        mlp_keys = jax.random.split(
            jax.random.PRNGKey(0), batch_size
        )  # PRNGKeyArray[Array, "2 2"]

        center_embed = center_embed + jax.vmap(lambda x, k: self.mlp(x, k))(
            center_embed, mlp_keys
        )  # Float[Array, "2 2"]

        return center_embed

    @beartype
    def create_message(
        self,
        center_embed: Float[Array, "2 2"],  # Shape: [batch_size, embed_dim]
        nodes: Float[Array, "2 node_dim=2"],  # Shape:  # Shape: [node, node_dim=2] Should be Float[Array, "node node_dim=2"]
        edge_index: Int[Array, "2 num_edges"],  # Shape: [2, num_edges] # This should be neighbors. For each node you should have a list of neighbors Adjacency matrix Bool[Array, "node node"]
        edge_attr: Float[Array, "num_edges edge_dim=2"],  # Shape: [num_edges, edge_dim] # Then we can ignore this one  because we have an adjmatrix
        rotate_mat: Float[Array, "batch_size 2 2"], # Shape: [batch_size, embed_dim, embed_dim]
    ): # -> Float[Array, "2 2 1"]
        # TODO replace x with node as that is a better name (List of nodes)
        # Rotation matrix is a [2,2] tensor
        # All 2,2 tensors
        # TODO switch to rel pos 2 neightbor
        center_embed_i = center_embed[edge_index[1]]  # target nodes, equivalent to center_embed_i in PyTorch Graph
        # Create a message funiton
        # First rotate the relative position by the rotation matrix

        x_j = nodes[edge_index[0]]  # Get source node features Float[Array, "2 2"]

        # Get center rotate mat
        center_rotate_mat = rotate_mat[edge_index[1]]  # Float[Array, "2 2 2"]

        # Rotate node features
        x_rotated = (
            rearrange(x_j, "n f -> n 1 f") @ center_rotate_mat
        )  # Float[Array, "2 1 2"]
        x_rotated = rearrange(x_rotated, "n 1 f -> n f")  # Float[Array, "2 2"]

        # Rotate edge features
        edge_rotated = (
            rearrange(edge_attr, "n f -> n 1 f") @ center_rotate_mat
        )  # Float[Array, "2 1 2"]

        edge_rotated = rearrange(edge_rotated, "n 1 f -> n f")  # Float[Array, "2 2"]

        # Compute neighbor embedding
        # nbr_embed = self._nbr_embed([x_rotated, edge_rotated])
        # Do vmap here
        # nbr_embed = jax.vmap(self._nbr_embed)([x_rotated, edge_rotated])
        # x_rotated: [batch_size, node_dim]

        nbr_embed = jax.vmap(lambda x, e: self._nbr_embed([x, e]))(
            x_rotated, edge_rotated
        )  # Float[Array, "2 2"]


        # print("nbr_embed.shape", nbr_embed.shape)
        # print("center_embed.shape", center_embed.shape)
        # print()
        # breakpoint()
        # Jax random key is different than torch so can expect slightly different results on the normalization
        ############################################################################################
        query = rearrange(
            jax.vmap(lambda x: self.lin_q(x))(center_embed_i),
            "n (h d) -> n h d",
            h=self.num_heads,
        )  # Float[Array, "2 2 1"]


        # Jax random key is different than torch so can expect slightly different results on the normalization

        # old way
        # key = rearrange(self.lin_k(nbr_embed), "n (h d) -> n h d", h=self.num_heads)

        key = rearrange(
            jax.vmap(lambda x: self.lin_k(x))(nbr_embed),
            "n (h d) -> n h d",
            h=self.num_heads,
        )  # Float[Array, "2 2 1"]

        # TODO check if this is correct with Marcell
        # Jax random key is different than torch so can expect slightly different results on the normalization
        value = rearrange(
            jax.vmap(lambda x: self.lin_v(x))(nbr_embed),
            "n (h d) -> n h d",
            h=self.num_heads,
        )  # Float[Array, "2 2 1"]

        scale = (self.embed_dim // self.num_heads) ** 0.5  # Float




        alpha = (query * key).sum(axis=-1) / scale  # Float[Array, "2 2"]

        # Do softmax
        alpha = jax.nn.softmax(alpha)  # Float[Array, "2 2"]

        # TODO Pass a key into this function as an additional random key
        alpha = self.attn_dropout(
            alpha, key=jax.random.PRNGKey(0)
        )  # Float[Array, "2 2"]

        messages = value * rearrange(alpha, "n h -> n h 1")  # Float[Array, "2 2 1"]
        ############################################################################################

        # aggregate
        # messages = reduce(
        #     messages, "n h d -> n d", "sum"
        # )  # This reduces to [b, d] # Float[Array, "2 1"]

        # This is done under the hood in torch the aggregation
        out = jax.ops.segment_sum(
            messages,
            edge_index[1],  # segment ids (target nodes)
            num_segments=nodes.shape[0]  # total number of nodes
        )  # Shape: [num_nodes, num_heads, head_dim]


        messages = rearrange(out, 'n h d -> n (h d)')  # Shape: [num_nodes, embed_dim]
        ## PROPAGATION IS DONE

        # Do equivilant of self.out_proj
        # Which is linear transformation to aggregated messaged

        messages = self.out_proj(messages)  # Float[Array, "2 2"]

        # Apply dropout
        # TODO pass a key into this function as an additional random key
        messages = self.proj_drop(
            messages, key=jax.random.PRNGKey(0)
        )  # Float[Array, "2 2"]

        # THen update logic to add to center_embed
        gate = jax.nn.sigmoid(
            self.lin_ih(messages) + self.lin_hh(center_embed)
        )  # Float[Array, "2 2"]

        messages = messages + gate * (
            self.lin_self(center_embed) - messages
        )  # Float[Array, "2 2"]

        print("EQX outputs", messages)
        return messages

        # Everything goes into message function
        # center_rotate_mat = rotate_mat[edge_index[1]]
