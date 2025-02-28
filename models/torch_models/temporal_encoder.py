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


from einops import rearrange, repeat
from beartype import beartype


from utils import print_array_type

# Add torch type signature

class TorchTemporalEncoder(nn.Module):

    def __init__(self,
                 historical_steps: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1) -> None:
        super(TorchTemporalEncoder, self).__init__()
        encoder_layer = TorchTemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout) # nn.TransformerEncoderLayer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(embed_dim))
        self.padding_token = nn.Parameter(torch.Tensor(historical_steps, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.Tensor(historical_steps + 1, 1, embed_dim))
        attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)
        self.register_buffer('attn_mask', attn_mask) # I am pretty sure i dont need to woryr about this
        nn.init.normal_(self.padding_token, mean=0., std=.02)
        nn.init.normal_(self.cls_token, mean=0., std=.02)
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        # self.apply(init_weights)

    def forward(self,
                x: torch.Tensor, # Shape: [historical_steps = 20, num_nodes = 2, xy = 2]
                padding_mask: torch.Tensor, # Shape: [xy=2, historical_steps = 20]
                ) -> torch.Tensor:

        padding_mask_transformed = rearrange(padding_mask, 'batch time -> time batch 1')
        x = torch.where(padding_mask_transformed, self.padding_token, x)
        # expand_cls_token = self.cls_token.expand(-1, x.shape[1], -1)
        expand_cls_token = repeat(self.cls_token, '1 1 d -> 1 b d', b=x.shape[1])
        x = torch.cat((x, expand_cls_token), dim=0) # Shape [historical_steps+1=21 nodes=2 xy=2] # Easier to just use torch.cat and add the cls token to the end
        x = x + self.pos_embed # Shape [historical_steps+1=21 nodes=2 xy=2]
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None, is_causal=False)
        return out[-1]  # [N, D]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class TorchTemporalEncoderLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1) -> None:
        super(TorchTemporalEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor, # Float[torch.Tensor, "historical_steps+1=21 num_nodes=2 xy=2"]
                src_mask: Optional[torch.Tensor] = None, # Float[torch.Tensor, "historical_steps+1=21 historical_steps+1=21"]
                src_key_padding_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        
        src_mask = kwargs.get('src_mask', None)

        x = src

        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Note the key padding mask is None anyways
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self,
                   x: torch.Tensor # Shape: [historical_steps+1=21 nodes=2 xy=2]
                   ) -> torch.Tensor: # Shape: [historical_steps+1=21 nodes=2 xy=2]
        x = self.linear2(self.dropout(F.relu_(self.linear1(x))))
        return self.dropout2(x)
    




