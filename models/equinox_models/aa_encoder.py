import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding


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
    parallel: bool
    dropout: float
    mlp: eqx.nn.Sequential
    bos_token: jnp.ndarray
    def __init__(self, historical_steps, node_dim, edge_dim, embed_dim, num_heads=8, dropout=0.1, parallel=False, *, key: Optional[PRNGKeyArray] = None):
        

        keys = jax.random.split(key, 12)

        print(f"keys shape: {keys.shape}")
        self.parallel = parallel
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self._center_embed = SingleInputEmbedding(node_dim, embed_dim, key=keys[0])
        self._nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], 
            out_channel=embed_dim,
            key=keys[1]
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

        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[9]),
            jax.nn.relu,
            eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[10]),
            eqx.nn.Dropout(dropout)
        ])

        # Key differences from PyTorch:
        # Uses jnp.ndarray instead of torch.Tensor
        # Initialization is explicit using JAX's random number generator
        # No need for nn.Parameter as Equinox automatically treats array attributes as parameters
        # Initialize BOS token with random values
        bos_key = jax.random.fold_in(keys[11], 0)
        self.bos_token = jax.random.normal(
            bos_key,
            shape=(self.historical_steps, self.embed_dim)
        ) * 0.02  # Scale factor similar to PyTorch's default initialization

    def __call__(self, x, edge_index, edge_attr, rotate_mat=None):

        pass
