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
    # attn_mask: jnp.ndarray

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
        # attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)


        
    def __call__(self, x: jnp.ndarray,
                  padding_mask: jnp.ndarray,
                    *, key: Optional[PRNGKeyArray] = None):
        
        padding_mask_transformed = rearrange(padding_mask, 'batch time -> time batch 1')
        x = jnp.where(padding_mask_transformed, self.padding_token, x)
        breakpoint()
        expand_cls_token = repeat(self.cls_token, '1 1 d -> 1 b d', b=x.shape[1])
        x = jnp.concatenate([x, expand_cls_token], axis=0)
        x = x + self.pos_embed
        out = self.transformer_encoder(src=x, mask=self.attn_mask, src_key_padding_mask=None, is_causal=False)
        return out[-1]



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


    def __call__(self, src: jnp.ndarray,
                  src_mask: Optional[jnp.ndarray] = None,
                  src_key_padding_mask: Optional[jnp.ndarray] = None,
                    *, 
                    key: Optional[jax.random.PRNGKey] = None,
                    ) -> jnp.ndarray:
        
        x = src

        x = x + self._sa_block(jax.vmap(self.norm1)(x), src_mask, src_key_padding_mask, key=key)
        
        print("Completed CALL")
        breakpoint()
        return x
    
    def _sa_block(self, x: jnp.ndarray,
                  attn_mask: Optional[jnp.ndarray] = None,
                  key_padding_mask: Optional[jnp.ndarray] = None,
                    *, 
                    key: Optional[jax.random.PRNGKey] = None,
                    ) -> jnp.ndarray:
        
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, key=key)
        return self.dropout1(x)
                
        


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
        self.layers = [
            eqx.tree_at(lambda x: x, encoder_layer, replace=True)
            for _ in range(num_layers)
        ]
        self.norm = norm

    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        *,
        key: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        # Split PRNG key for each layer if provided
        keys = None if key is None else jax.random.split(key, len(self.layers))
        
        # Pass through each layer
        for layer, layer_key in zip(self.layers, keys or [None] * len(self.layers)):
            x = layer(x, mask=mask, key=layer_key)
            
        # Apply final normalization if provided
        if self.norm is not None:
            x = self.norm(x)
            
        return x