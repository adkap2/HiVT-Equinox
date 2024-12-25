# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
from utils.torch.torch_utils import DistanceDropEdge


# from models import MultipleInputEmbedding
from .embedding import TorchSingleInputEmbedding, TorchMultipleInputEmbedding
from .aa_encoder import TorchAAEncoder
from .temporal_encoder import TorchTemporalEncoder
# from utils import DistanceDropEdge
# from utils import TemporalData
# from utils import init_weights


class LocalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 local_radius: float = 50,
                 parallel: bool = False) -> None:
        super(LocalEncoder, self).__init__()
        self.historical_steps = historical_steps
        # self.parallel = parallel

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = TorchAAEncoder(historical_steps=historical_steps,
                                    node_dim=node_dim,
                                    edge_dim=edge_dim,
                                    embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    dropout=dropout,)
        self.temporal_encoder = TorchTemporalEncoder(historical_steps=historical_steps,
                                                embed_dim=embed_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                num_layers=num_temporal_layers)
        # self.al_encoder = ALEncoder(node_dim=node_dim,
        #                             edge_dim=edge_dim,
        #                             embed_dim=embed_dim,
        #                             num_heads=num_heads,
        #                             dropout=dropout)

    def forward(self,
                 data# :  TemporalData
                 )-> torch.Tensor:
        for t in range(self.historical_steps):
            data[f'edge_index_{t}'], _ = subgraph(subset=~data['padding_mask'][:, t], edge_index=data['edge_index'])
            data[f'edge_attr_{t}'] = \
                data['positions'][data[f'edge_index_{t}'][0], t] - data['positions'][data[f'edge_index_{t}'][1], t]

        # Should be temporal data object but for now we will use dictionary
        # print("Data at edge index at t = 0")

        # print("edge index and edge attr shape at t = 0")
        # print(data['edge_index_0'].shape)
        # print(data['edge_attr_0'].shape)
        # breakpoint()


        out = [None] * self.historical_steps
        for t in range(self.historical_steps):
            edge_index, edge_attr = self.drop_edge(data[f'edge_index_{t}'], data[f'edge_attr_{t}'])

            # print("TORCH edge_index.shape", edge_index.shape)
            # print("TORCH edge_attr.shape", edge_attr.shape)
            # print("TORCH x.shape", data['x'][:, t].shape)
            # print("TORCH rotate_mat.shape", data['rotate_mat'].shape)
            # print("TORCH bos_mask.shape", data['bos_mask'][:, t].shape)
            # breakpoint()   


            out[t] = self.aa_encoder(x=data['x'][:, t], t=t, edge_index=edge_index, edge_attr=edge_attr,
                                        bos_mask=data['bos_mask'][:, t], rotate_mat=data['rotate_mat'])


        # print("Stackinh out")
        # breakpoint()
        out = torch.stack(out)  # [T, N, D]
        # print("Preparing for temporal encoder")
        # breakpoint()
        out = self.temporal_encoder(x=out, padding_mask=data['padding_mask'][:, : self.historical_steps])
        # edge_index, edge_attr = self.drop_edge(data['lane_actor_index'], data['lane_actor_vectors'])
        # out = self.al_encoder(x=(data['lane_vectors'], out), edge_index=edge_index, edge_attr=edge_attr,
        #                       is_intersections=data['is_intersections'], turn_directions=data['turn_directions'],
        #                       traffic_controls=data['traffic_controls'], rotate_mat=data['rotate_mat'])
        return out





# class TemporalEncoder(nn.Module):

#     def __init__(self,
#                  historical_steps: int,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  num_layers: int = 4,
#                  dropout: float = 0.1) -> None:
#         super(TemporalEncoder, self).__init__()
#         encoder_layer = TemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
#                                                          norm=nn.LayerNorm(embed_dim))
#         self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
#         self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
#         attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
#         self.register_buffer('attn_mask', attn_mask)
#         nn.init.normal_(self.padding_token, mean=0., std=.02)
#         nn.init.normal_(self.cls_token, mean=0., std=.02)
#         nn.init.normal_(self.pos_embed, mean=0., std=.02)
#         self.apply(init_weights)

#     def forward(self,
#                 x: torch.Tensor,
#                 padding_mask: torch.Tensor) -> torch.Tensor:
#         x = torch.where(padding_mask.t().unsqueeze(-1), self.padding_token, x)
#         expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
#         x = torch.cat((x, expand_cls_token), dim=0)
#         x = x + self.pos_embed
#         out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None)
#         return out[-1]  # [N, D]

#     @staticmethod
#     def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
#         mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask


# class TemporalEncoderLayer(nn.Module):

#     def __init__(self,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  dropout: float = 0.1) -> None:
#         super(TemporalEncoderLayer, self).__init__()
#         self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
#         self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#     def forward(self,
#                 src: torch.Tensor,
#                 src_mask: Optional[torch.Tensor] = None,
#                 src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         x = src
#         x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#         x = x + self._ff_block(self.norm2(x))
#         return x

#     def _sa_block(self,
#                   x: torch.Tensor,
#                   attn_mask: Optional[torch.Tensor],
#                   key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
#         x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
#         return self.dropout1(x)

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
#         return self.dropout2(x)


# class ALEncoder(MessagePassing):

#     def __init__(self,
#                  node_dim: int,
#                  edge_dim: int,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  dropout: float = 0.1,
#                  **kwargs) -> None:
#         super(ALEncoder, self).__init__(aggr='add', node_dim=0, **kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads

#         self.lane_embed = MultipleInputEmbedding(in_channels=[node_dim, edge_dim], out_channel=embed_dim)
#         self.lin_q = nn.Linear(embed_dim, embed_dim)
#         self.lin_k = nn.Linear(embed_dim, embed_dim)
#         self.lin_v = nn.Linear(embed_dim, embed_dim)
#         self.lin_self = nn.Linear(embed_dim, embed_dim)
#         self.attn_drop = nn.Dropout(dropout)
#         self.lin_ih = nn.Linear(embed_dim, embed_dim)
#         self.lin_hh = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout))
#         self.is_intersection_embed = nn.Parameter(torch.Tensor(2, embed_dim))
#         self.turn_direction_embed = nn.Parameter(torch.Tensor(3, embed_dim))
#         self.traffic_control_embed = nn.Parameter(torch.Tensor(2, embed_dim))
#         nn.init.normal_(self.is_intersection_embed, mean=0., std=.02)
#         nn.init.normal_(self.turn_direction_embed, mean=0., std=.02)
#         nn.init.normal_(self.traffic_control_embed, mean=0., std=.02)
#         self.apply(init_weights)

#     def forward(self,
#                 x: Tuple[torch.Tensor, torch.Tensor],
#                 edge_index: Adj,
#                 edge_attr: torch.Tensor,
#                 is_intersections: torch.Tensor,
#                 turn_directions: torch.Tensor,
#                 traffic_controls: torch.Tensor,
#                 rotate_mat: Optional[torch.Tensor] = None,
#                 size: Size = None) -> torch.Tensor:
#         x_lane, x_actor = x
#         is_intersections = is_intersections.long()
#         turn_directions = turn_directions.long()
#         traffic_controls = traffic_controls.long()
#         x_actor = x_actor + self._mha_block(self.norm1(x_actor), x_lane, edge_index, edge_attr, is_intersections,
#                                             turn_directions, traffic_controls, rotate_mat, size)
#         x_actor = x_actor + self._ff_block(self.norm2(x_actor))
#         return x_actor

#     def message(self,
#                 edge_index: Adj,
#                 x_i: torch.Tensor,
#                 x_j: torch.Tensor,
#                 edge_attr: torch.Tensor,
#                 is_intersections_j,
#                 turn_directions_j,
#                 traffic_controls_j,
#                 rotate_mat: Optional[torch.Tensor],
#                 index: torch.Tensor,
#                 ptr: OptTensor,
#                 size_i: Optional[int]) -> torch.Tensor:
#         if rotate_mat is None:
#             x_j = self.lane_embed([x_j, edge_attr],
#                                   [self.is_intersection_embed[is_intersections_j],
#                                    self.turn_direction_embed[turn_directions_j],
#                                    self.traffic_control_embed[traffic_controls_j]])
#         else:
#             rotate_mat = rotate_mat[edge_index[1]]
#             x_j = self.lane_embed([torch.bmm(x_j.unsqueeze(-2), rotate_mat).squeeze(-2),
#                                    torch.bmm(edge_attr.unsqueeze(-2), rotate_mat).squeeze(-2)],
#                                   [self.is_intersection_embed[is_intersections_j],
#                                    self.turn_direction_embed[turn_directions_j],
#                                    self.traffic_control_embed[traffic_controls_j]])
#         query = self.lin_q(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         key = self.lin_k(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         value = self.lin_v(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         scale = (self.embed_dim // self.num_heads) ** 0.5
#         alpha = (query * key).sum(dim=-1) / scale
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = self.attn_drop(alpha)
#         return value * alpha.unsqueeze(-1)

#     def update(self,
#                inputs: torch.Tensor,
#                x: torch.Tensor) -> torch.Tensor:
#         x_actor = x[1]
#         inputs = inputs.view(-1, self.embed_dim)
#         gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x_actor))
#         return inputs + gate * (self.lin_self(x_actor) - inputs)

#     def _mha_block(self,
#                    x_actor: torch.Tensor,
#                    x_lane: torch.Tensor,
#                    edge_index: Adj,
#                    edge_attr: torch.Tensor,
#                    is_intersections: torch.Tensor,
#                    turn_directions: torch.Tensor,
#                    traffic_controls: torch.Tensor,
#                    rotate_mat: Optional[torch.Tensor],
#                    size: Size) -> torch.Tensor:
#         x_actor = self.out_proj(self.propagate(edge_index=edge_index, x=(x_lane, x_actor), edge_attr=edge_attr,
#                                                is_intersections=is_intersections, turn_directions=turn_directions,
#                                                traffic_controls=traffic_controls, rotate_mat=rotate_mat, size=size))
#         return self.proj_drop(x_actor)

#     def _ff_block(self, x_actor: torch.Tensor) -> torch.Tensor:
#         return self.mlp(x_actor)
