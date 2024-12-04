from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph


from einops import rearrange
from beartype import beartype

from models.torch_models.embedding import (
    TorchSingleInputEmbedding,
    TorchMultipleInputEmbedding,
)

from utils import print_array_type

# Add torch type signature


class TorchAAEncoder(MessagePassing):

    @beartype
    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        parallel: bool = False,
        **kwargs,
    ) -> None:
        super(TorchAAEncoder, self).__init__(
            aggr="add", node_dim=0, **kwargs
        )  # Note they intiiakize with add aggregation.
        # For add aggregation, the message passing is done in the update function
        # # Node div indiciates which axis to propogate
        # # In this case, we want to propogate along the node axis
        # By default it's batch, node, channel
        # i is fixed and j is the neighbor
        # compute the messages and then sum them up
        # then apply the update function
        # Ours is better to be denise. Compute the messages for every node and then sum them up
        # Calculate the same operation for everything

        self.historical_steps = historical_steps  # int
        self.embed_dim = embed_dim  # int
        self.num_heads = num_heads  # int
        self.parallel = parallel  # bool

        self.center_embed = TorchSingleInputEmbedding(
            in_channel=node_dim, out_channel=embed_dim
        )  # TorchSingleInputEmbedding
        self.nbr_embed = TorchMultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
        )  # TorchMultipleInputEmbedding
        self.lin_q = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.lin_k = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.lin_v = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.lin_self = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.attn_drop = nn.Dropout(dropout)  # nn.Dropout
        self.lin_ih = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.lin_hh = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # nn.Linear
        self.proj_drop = nn.Dropout(dropout)  # nn.Dropout
        self.norm1 = nn.LayerNorm(embed_dim)  # nn.LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)  # nn.LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # nn.Linear -> 4x embed_dim
            nn.ReLU(inplace=True),  # nn.ReLU
            nn.Dropout(dropout),  # nn.Dropout
            nn.Linear(embed_dim * 4, embed_dim),  # nn.Linear -> 4x embed_dim
            nn.Dropout(dropout),  # nn.Dropout
        )
        self.bos_token = nn.Parameter(
            torch.Tensor(historical_steps, embed_dim)
        )  # nn.Parameter
        nn.init.normal_(self.bos_token, mean=0.0, std=0.02)
        # TODO: Add initialization for the weights
        # self.apply(init_weights)

    @beartype
    def forward(
        self,
        x: torch.Tensor,  # Shape: [batch_size, node_dim] -> Float[Tensor, "2 2"]
        t: Optional[int],
        edge_index: torch.Tensor,  # Assuming shape -> Int[Tensor, "2 2"]
        edge_attr: torch.Tensor,  # Assuming shape [num_edges, edge_dim] -> Float[Tensor, "2 2"]
        bos_mask: torch.Tensor,  # Assuming shape [batch_size] -> Bool[Tensor, "2"]
        rotate_mat: Optional[
            torch.Tensor
        ] = None,  # Assuming shape [batch_size, embed_dim, embed_dim] -> Float[Tensor, "2 2 2"]
        size: Size = None,
    ) -> (
        torch.Tensor
    ):  # Assuming output shape [batch, embed_dim] -> Float[Tensor, "2 2"]

        # x is list of 2D points

        if rotate_mat is None:
            # We want this one
            center_embed = self.center_embed(x)  #
        else:
            # center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2)) # Call forward
            # Replace this with einops
            # Rotation matrix is a 2x2x2 matrix
            x_rotated = rearrange(x, "n f -> n 1 f") @ rotate_mat
            x_rotated = rearrange(x_rotated, "n 1 f -> n f")  # Float[Tensor, "2 2"]

            center_embed = self.center_embed(x_rotated)  # Float[Tensor, "2 2"]

        # Einops
        # Using einops instead of unsqueeze
        bos_mask = rearrange(bos_mask, "n -> n 1")  # Bool[Tensor, "2 1"]

        center_embed = torch.where(
            bos_mask, self.bos_token[t], center_embed
        )  # Float[Tensor, "2 2"]

        center_embed = center_embed + self._mha_block(
            self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size
        )  # Apply mha block to the center embed this should be the message passing Float[Tensor, "2 2"]

        center_embed = center_embed + self._ff_block(
            self.norm2(center_embed)
        )  # Float[Tensor, "2 2"]

        return center_embed

        # Think about inputs of graph and outputs of the graph
        # Just mechanical translation this module for now

    @beartype
    def message(
        self,
        edge_index: Adj,  # Shape: [2, num_edges] -> Int[Tensor, "2 2"]
        center_embed_i: torch.Tensor,  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2"]
        x_j: torch.Tensor,  # Shape: [num_edges, node_dim] -> Float[Tensor, "2 2"]
        edge_attr: torch.Tensor,  # Shape: [num_edges, edge_dim] -> Float[Tensor, "2 2"]
        rotate_mat: Optional[
            torch.Tensor
        ],  # Shape: [batch_size, embed_dim, embed_dim] -> Float[Tensor, "2 2 2"]
        index: torch.Tensor,  # Shape: [num_edges] -> Int[Tensor, "2"]
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2 1"]

        if rotate_mat is None:
            nbr_embed = self.nbr_embed(
                [x_j, edge_attr]
            )  # Call forward (Neighbor embedding)
        else:

            center_rotate_mat = rotate_mat[edge_index[1]]  # Float[Tensor, "2 2 2"]

            # Rotate node and edge features using einops
            x_rotated = rearrange(x_j, "n f -> n 1 f") @ center_rotate_mat
            x_rotated = rearrange(x_rotated, "n 1 f -> n f")  # Float[Tensor, "2 2"]

            edge_rotated = rearrange(edge_attr, "n f -> n 1 f") @ center_rotate_mat
            edge_rotated = rearrange(
                edge_rotated, "n 1 f -> n f"
            )  # Float[Tensor, "2 2"]

            nbr_embed = self.nbr_embed(
                [x_rotated, edge_rotated]
            )  # Float[Tensor, "2 2"]

        # Can replace all this with MultiHeadAttention
        query = rearrange(
            self.lin_q(center_embed_i), "n (h d) -> n h d", h=self.num_heads
        )  # Float[Tensor, "2 2 1"]

        key = rearrange(
            self.lin_k(nbr_embed), "n (h d) -> n h d", h=self.num_heads
        )  # Float[Tensor, "2 2 1"]

        # print("TORCH key", key)
        value = rearrange(
            self.lin_v(nbr_embed), "n (h d) -> n h d", h=self.num_heads
        )  # Float[Tensor, "2 2 1"]

        scale = (self.embed_dim // self.num_heads) ** 0.5

        alpha = (query * key).sum(dim=-1) / scale  # Float[Tensor, "2 2"]

        alpha = softmax(
            alpha, index, ptr, size_i
        )  # Size is None which is the same as doing softwmax over full array Float[Tensor, "2 2"]

        alpha = self.attn_drop(alpha)  # Float[Tensor, "2 2"]

        messages = value * rearrange(alpha, "n h -> n h 1")  # Float[Tensor, "2 2 1"]
        return messages

    @beartype
    def update(self, inputs: torch.Tensor, center_embed: torch.Tensor) -> torch.Tensor:
        inputs = rearrange(inputs, "b d 1 -> b d")  # Float[Tensor, "2 2"]

        # Their update is just a non linear sigmoid
        gate = torch.sigmoid(
            self.lin_ih(inputs) + self.lin_hh(center_embed)
        )  # Float[Tensor, "2 2"]

        outputs = inputs + gate * (
            self.lin_self(center_embed) - inputs
        )  # Float[Tensor, "2 2"]
        return outputs

    @beartype
    def _mha_block(
        self,
        center_embed: torch.Tensor,  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2"]
        x: torch.Tensor,  # Shape: [num_edges, node_dim] -> Float[Tensor, "2 2"]
        edge_index: Adj,  # Shape: [2, num_edges] -> Int[Tensor, "2 2"]
        edge_attr: torch.Tensor,  # Shape: [num_edges, edge_dim] -> Float[Tensor, "2 2"]
        rotate_mat: Optional[
            torch.Tensor
        ],  # Shape: [batch_size, embed_dim, embed_dim] -> Float[Tensor, "2 2 2"]
        size: Size,
    ) -> torch.Tensor:  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2"]

        center_embed = self.out_proj(
            self.propagate(
                edge_index=edge_index,  # Shape: [2, num_edges] -> [2, 2]
                x=x,  # Shape: [num_edges, node_dim]
                center_embed=center_embed,  # Shape: [batch_size, embed_dim]
                edge_attr=edge_attr,  # Shape: [num_edges, edge_dim]
                rotate_mat=rotate_mat,  # Shape: [batch_size, embed_dim, embed_dim]
                size=size,
            )
        )  # Float[Tensor, "2 2"]

        outputs = self.proj_drop(center_embed)  # Float[Tensor, "2 2"]

        return outputs

    @beartype
    def _ff_block(
        self, x: torch.Tensor  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2"]
    ) -> torch.Tensor:  # Shape: [batch_size, embed_dim] -> Float[Tensor, "2 2"]

        outputs = self.mlp(x)  # Float[Tensor, "2 2"]
        return outputs
