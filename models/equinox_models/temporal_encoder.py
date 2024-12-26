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

# Add jax type signature to inputs and outputs


class TransformerEncoder(eqx.Module):
    # ... existing TransformerEncoder implementation ...
    pass

class EquinoxTemporalEncoderLayer(eqx.Module):
    pass

class TemporalEncoder(eqx.Module):
    historical_steps: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float

    transformer_encoder: TransformerEncoder
    encoder_layer: eqx.Module
    padding_token: jnp.ndarray
    cls_token: jnp.ndarray
    pos_embed: jnp.ndarray

    attn_mask: jnp.ndarray

    @beartype
    def __init__(self, historical_steps: int, embed_dim: int, num_heads: int, num_layers: int, dropout: float, *, key: Optional[PRNGKeyArray] = None):
        keys = jax.random.split(key, 4)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout


        self.encoder_layer = EquinoxTemporalEncoderLayer(embed_dim=self.embed_dim, num_heads=self.num_heads, dropout=self.dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=self.num_layers, norm=eqx.nn.LayerNorm(self.embed_dim))
        self.padding_token = jax.random.normal(keys[0], (historical_steps, 1, self.embed_dim))
        self.cls_token = jax.random.normal(keys[1], (1, 1, self.embed_dim))
        self.pos_embed = jax.random.normal(keys[2], (historical_steps + 1, 1, self.embed_dim))
        self.attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)


        
    def __call__(self, x: jnp.ndarray, # [historical_steps=20, num_nodes=2, xy=2]
                  padding_mask: jnp.ndarray,
                    *, key: Optional[PRNGKeyArray] = None):
        padding_mask_transformed = rearrange(padding_mask, 'batch time -> time batch 1')
        x = jnp.where(padding_mask_transformed, self.padding_token, x)
        expand_cls_token = repeat(self.cls_token, '1 1 d -> 1 b d', b=x.shape[1])
        x = jnp.concatenate([x, expand_cls_token], axis=0)
        x = x + self.pos_embed
        # out = self.transformer_encoder(x=x, mask=self.attn_mask, key=key, src_key_padding_mask=None)
        # Vmap the entire transformer encoder
        vmapped_transformer_encoder = jax.vmap(
            lambda x: self.transformer_encoder(
                x=x,
                mask=self.attn_mask,
                key=key,
                src_key_padding_mask=None
            ),
            in_axes=1,  # vmap over second dimension (num_nodes)
            out_axes=1
        )
        out = vmapped_transformer_encoder(x)
        return out[-1]
    
    def generate_square_subsequent_mask(self, seq_len: int) -> jnp.ndarray:
        # Create initial mask where 1s are in upper triangle (including diagonal)
        mask = (jnp.triu(jnp.ones((seq_len, seq_len))) == 1).transpose(0, 1)
        
        # Convert to float and replace:
        # - False (0) becomes -inf
        # - True (1) becomes 0.0
        mask = jnp.where(mask, 0.0, float('-inf'))
        
        return mask



class EquinoxTemporalEncoderLayer(eqx.Module):
    embed_dim: int
    num_heads: int
    dropout: float
    dropout0: eqx.nn.Dropout
    self_attn: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, *, key: Optional[PRNGKeyArray] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 2)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.self_attn = eqx.nn.MultiheadAttention(num_heads=self.num_heads, query_size=self.embed_dim, key=keys[0])
        self.linear1 = eqx.nn.Linear(self.embed_dim, self.embed_dim * 4, key=keys[1])
        self.dropout0 = eqx.nn.Dropout(self.dropout)
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[2])
        self.norm1 = eqx.nn.LayerNorm(self.embed_dim)
        self.norm2 = eqx.nn.LayerNorm(self.embed_dim)
        self.dropout1 = eqx.nn.Dropout(self.dropout)
        self.dropout2 = eqx.nn.Dropout(self.dropout)


    def __call__(self, src: jnp.ndarray, # [historical_steps+1=21, xy=2]
                  src_mask: Optional[jnp.ndarray] = None, # [historical_steps+1=21, historical_steps+1=21]
                  src_key_padding_mask: Optional[jnp.ndarray] = None,
                    *, 
                    key: Optional[jax.random.PRNGKey] = None,
                    ) -> jnp.ndarray:
        
        x = src
        # Vmap LayerNorm over sequence dimension
        vmapped_norm1 = jax.vmap(self.norm1)
        vmapped_norm2 = jax.vmap(self.norm2)
        x = x + self._sa_block(vmapped_norm1(x), attn_mask=src_mask, key_padding_mask=src_key_padding_mask, key=key)
        x = x + self._ff_block(vmapped_norm2(x), key=key)

        return x
    
    def _sa_block(self, x: jnp.ndarray,
                  attn_mask: Optional[jnp.ndarray] = None,
                  key_padding_mask: Optional[jnp.ndarray] = None,
                    *, 
                    key: Optional[jax.random.PRNGKey] = None,
                    ) -> jnp.ndarray:

        keys = jax.random.split(key, 2)


        x = self.self_attn(query=x,
                            key_=x,
                              value=x,
                                mask=attn_mask, 
                                key=keys[0],
                                )
        return self.dropout1(x, key=keys[1])
    
    def _ff_block(self, x: jnp.ndarray, # [historical_steps+1=21, xy=2]
                  key: Optional[jax.random.PRNGKey] = None,
                    ) -> jnp.ndarray:


        # Split PRNG key for the two dropout operations
        if key is not None:
            key1, key2 = jax.random.split(key)
        else:
            key1 = key2 = None

        # Vmap the linear layers over sequence dimension
        vmapped_linear1 = jax.vmap(self.linear1)
        vmapped_linear2 = jax.vmap(self.linear2)


        x = vmapped_linear1(x)  # [historical_steps+1=21, embed_dim*4=8]
        x = jax.nn.relu(x)
        x = self.dropout0(x, key=key1)
        x = vmapped_linear2(x)  # [21, 2, 2]
        x = self.dropout2(x, key=key2)
        
        return x
                
        


class TransformerEncoder(eqx.Module):
    """TransformerEncoder is a stack of N encoder layers."""
    
    layers: List[eqx.Module]
    norm: Optional[eqx.nn.LayerNorm]
    encoder_layer: eqx.Module
    num_layers: int


    
    def __init__(
        self,
        encoder_layer: eqx.Module,
        num_layers: int,
        norm: Optional[eqx.nn.LayerNorm] = None,
    ):
        """
        Args:
            encoder_layer: Single transformer encoder layer
            num_layers: Number of times to stack the encoder layer
            norm: Optional layer normalization
        """

        self.num_layers = num_layers
        self.encoder_layer = encoder_layer
        # Create copies of the encoder layer
        self.layers = [copy.deepcopy(encoder_layer) for _ in range(num_layers)]

        self.norm = norm

    def __call__(
        self,
        x: jnp.ndarray, # [historical_steps=20, xy=2]
        mask: Optional[jnp.ndarray] = None,
        src_key_padding_mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        
        if key is None:
            key = jax.random.PRNGKey(0)
        # Split PRNG key for each layer if provided
        keys = jax.random.split(key, len(self.layers))
        output = x

        def f(x, layer, key):
            return layer(x, 
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask,
                         key=key)
        

        for layer, key in zip(self.layers, keys):
            output = f(output, layer, key=key)

        vmapped_norm = jax.vmap(self.norm)

        # Apply final normalization if provided
        if self.norm is not None:
            x = vmapped_norm(x)
            
        return x