import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool

from einops import rearrange, repeat, reduce

from beartype import beartype

@beartype
class TransformerEncoder(eqx.Module):
    pass

@beartype
class EquinoxTemporalEncoderLayer(eqx.Module):
    pass


@beartype
class TemporalEncoder(eqx.Module):
    historical_steps: int
    embed_dim: int
    num_heads: int
    num_layers: int
    dropout: float

    transformer_encoder: TransformerEncoder
    padding_token: jnp.ndarray
    cls_token: jnp.ndarray
    pos_embed: jnp.ndarray

    attn_mask: jnp.ndarray

    def __init__(
        self,
        historical_steps: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 4)
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.transformer_encoder = TransformerEncoder(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            key=keys[0],
            num_layers=self.num_layers,
        )
        self.padding_token = jax.random.normal(keys[1], (1, self.embed_dim))

        # TODO make this an embedding using the eqx.nn.Embedding
        self.cls_token = jax.random.normal(keys[2], (1, self.embed_dim))
        # TODO make this an embedding using the eqx.nn.Embedding
        self.pos_embed = jax.random.normal(keys[3], (historical_steps + 1, self.embed_dim))
        self.attn_mask = self.generate_square_subsequent_mask(historical_steps + 1)

        # TODO Add INIT weights

    def __call__(
        self,
        x: Float[Array, "historical_steps=20 d=2"],
        padding_mask: Bool[Array, "historical_steps=20"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "d=2"]:

        # Add the padding token to the beginning of the sequence
        x = jnp.vstack([x, self.cls_token])  # Will be (21, 2)
        x = x + self.pos_embed  # [historical_steps+1=21, hidden_dim]

        padding_mask = jnp.pad(
            ~padding_mask, ((0, 1),), constant_values=1
        )  # Add cls token -> [num_nodes, 20]

        # Compute new_mask
        new_mask = jnp.logical_and(
            jnp.outer(padding_mask, padding_mask),
            self.attn_mask,
        )
        out = self.transformer_encoder(x=x, key=key, mask=new_mask)
        return out[-1]

    @staticmethod
    def generate_square_subsequent_mask(seq_len: int) -> jnp.ndarray:
        # Create initial mask where 1s are in upper triangle (including diagonal)
        mask = (jnp.triu(jnp.ones((seq_len, seq_len))) == 1).transpose(0, 1)
        # Convert to float and replace:
        # - False (0) becomes -inf
        # - True (1) becomes 0.0
        mask = jnp.where(mask, 0.0, float("-inf"))
        return mask


@beartype
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

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, *, key: PRNGKeyArray
    ):

        keys = jax.random.split(key, 2)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=self.num_heads, query_size=self.embed_dim, key=keys[0]
        )
        self.linear1 = eqx.nn.Linear(self.embed_dim, self.embed_dim * 4, key=keys[1])
        self.dropout0 = eqx.nn.Dropout(self.dropout)
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[2])
        self.norm1 = eqx.nn.LayerNorm(self.embed_dim)
        self.norm2 = eqx.nn.LayerNorm(self.embed_dim)
        self.dropout1 = eqx.nn.Dropout(self.dropout)
        self.dropout2 = eqx.nn.Dropout(self.dropout)

    def __call__(
        self,
        src: Float[Array, "t=20+1 d=2"],
        src_mask: Bool[Array, "t=20+1 t=20+1"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "t=20+1 d=2"]:
        key1, key2 = jax.random.split(key, 2)
        x = src
        # Vmap LayerNorm over sequence dimension
        vmapped_norm1 = jax.vmap(self.norm1)
        vmapped_norm2 = jax.vmap(self.norm2)
        # TODO write this as inline where its one function
        x = vmapped_norm1(x)

        x = x + self.self_attn(query=x, key_=x, value=x, mask=src_mask, key=key1)
        x = x + self._ff_block(vmapped_norm2(x), key=key2)

        return x

    def _ff_block(
        self,
        x: Float[Array, "t=20+1 d=2"],
        key: PRNGKeyArray,
    ) -> Float[Array, "t=20+1 d=2"]:

        # Split PRNG key for the two dropout operations
        key1, key2 = jax.random.split(key)

        # Vmap the linear layers over sequence dimension
        vmapped_linear1 = jax.vmap(self.linear1)
        vmapped_linear2 = jax.vmap(self.linear2)

        x = vmapped_linear1(x)
        x = jax.nn.relu(x)
        x = self.dropout0(x, key=key1)
        x = vmapped_linear2(x)
        x = self.dropout2(x, key=key2)

        return x


@beartype
class TransformerEncoder(eqx.Module):
    """TransformerEncoder is a stack of N encoder layers."""

    layers: List[eqx.Module]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        *,
        key: PRNGKeyArray,
        num_layers: int,
    ):
        """
        Args:
            embed_dim: The dimension of the input embeddings
            num_heads: The number of attention heads
            dropout: The dropout rate
            key: The PRNG key for initializing the layers
            num_layers: The number of layers to stack
        """

        # Create copies of the encoder layer
        self.layers = []
        for key in jax.random.split(key, num_layers):
            self.layers.append(EquinoxTemporalEncoderLayer(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, key=key))

    def __call__(
        self,
        x: Float[Array, "t=20+1 d=2"],
        mask: Bool[Array, "t=20+1 t=20+1"],
        *,
        key: PRNGKeyArray,
    ) -> Float[Array, "t=20+1 d=2"]:

        # Split PRNG key for each layer if provided
        keys = jax.random.split(key, len(self.layers))
        output = x

        def f(x, layer, key):
            return layer(x, src_mask=mask, key=key)

        # TODO convert to a scan
        for layer, key in zip(self.layers, keys):
            output = f(output, layer, key=key)

        return output
