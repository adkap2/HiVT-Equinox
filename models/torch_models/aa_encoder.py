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

        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.parallel = parallel

        self.center_embed = TorchSingleInputEmbedding(
            in_channel=node_dim, out_channel=embed_dim
        )
        self.nbr_embed = TorchMultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], out_channel=embed_dim
        )
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
        nn.init.normal_(self.bos_token, mean=0.0, std=0.02)
        # TODO: Add initialization for the weights
        # self.apply(init_weights)

    @beartype
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[int],
        edge_index: Adj,
        edge_attr: torch.Tensor,
        bos_mask: torch.Tensor,
        rotate_mat: Optional[torch.Tensor] = None,
        size: Size = None,
    ) -> torch.Tensor:

        # print(f"[PyTorch] Input x shape: {x.shape}")
        if rotate_mat is None:
            # print("x.shape", x.shape)
            # We want this one
            center_embed = self.center_embed(x)  # Call forward
        else:
            # center_embed = self.center_embed(torch.bmm(x.unsqueeze(-2), rotate_mat).squeeze(-2)) # Call forward
            # Replace this with einops
            x_rotated = rearrange(x, "n f -> n 1 f") @ rotate_mat
            x_rotated = rearrange(x_rotated, "n 1 f -> n f")
            center_embed = self.center_embed(x_rotated)

        # center_embed = torch.where(bos_mask.unsqueeze(-1), self.bos_token[t], center_embed) # Apply bos mask to the center embed
        # Einops
        # Using einops instead of unsqueeze
        bos_mask = rearrange(bos_mask, "n -> n 1")

        center_embed = torch.where(bos_mask, self.bos_token[t], center_embed)

        # print(f"[PyTorch] center_embed shape: {center_embed.shape}")
        # print(f"[PyTorch] center_embed first few values: {center_embed[0, :5]}")

        center_embed = center_embed + self._mha_block(
            self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size
        )  # Apply mha block to the center embed this should be the message passing

        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

        # Think about inputs of graph and outputs of the graph
        # Just mechanical translation this module for now
    @beartype
    def message(
        self,
        edge_index: Adj,
        center_embed_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> torch.Tensor:

        if rotate_mat is None:
            nbr_embed = self.nbr_embed(
                [x_j, edge_attr]
            )  # Call forward (Neighbor embedding)
        else:
            # if self.parallel:
            #     center_rotate_mat = rotate_mat.repeat(self.historical_steps, 1, 1)[edge_index[1]]
            # else:
            # print(f"[PyTorch] rotate_mat shape: {rotate_mat.shape}")
            center_rotate_mat = rotate_mat[edge_index[1]]
            # print(f"[PyTorch] center_rotate_mat shape: {center_rotate_mat.shape}")

            # Rotate node and edge features using einops
            x_rotated = rearrange(x_j, "n f -> n 1 f") @ center_rotate_mat
            x_rotated = rearrange(x_rotated, "n 1 f -> n f")

            edge_rotated = rearrange(edge_attr, "n f -> n 1 f") @ center_rotate_mat
            edge_rotated = rearrange(edge_rotated, "n 1 f -> n f")

            nbr_embed = self.nbr_embed([x_rotated, edge_rotated])
        # Can replace all this with MultiHeadAttention
        query = rearrange(
            self.lin_q(center_embed_i), "n (h d) -> n h d", h=self.num_heads
        )

        key = rearrange(self.lin_k(nbr_embed), "n (h d) -> n h d", h=self.num_heads)

        value = rearrange(self.lin_v(nbr_embed), "n (h d) -> n h d", h=self.num_heads)

        scale = (self.embed_dim // self.num_heads) ** 0.5

        alpha = (query * key).sum(dim=-1) / scale

        alpha = softmax(
            alpha, index, ptr, size_i
        )  # Size is None which is the same as doing softwmax over full array

        alpha = self.attn_drop(alpha)
        return value * rearrange(alpha, "n h -> n h 1")

    @beartype
    def update(self, inputs: torch.Tensor, center_embed: torch.Tensor) -> torch.Tensor:
        inputs = rearrange(inputs, "b d 1 -> b d")

        # Their update is just a non linear sigmoid
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))

        return inputs + gate * (self.lin_self(center_embed) - inputs)

    @beartype
    def _mha_block(
        self,
        center_embed: torch.Tensor,
        x: torch.Tensor,
        edge_index: Adj,
        edge_attr: torch.Tensor,
        rotate_mat: Optional[torch.Tensor],
        size: Size,
    ) -> torch.Tensor:

        center_embed = self.out_proj(
            self.propagate(
                edge_index=edge_index,
                x=x,
                center_embed=center_embed,
                edge_attr=edge_attr,
                rotate_mat=rotate_mat,
                size=size,
            )
        )
        return self.proj_drop(center_embed)

    @beartype
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
