from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int
from beartype import beartype
from .aa_encoder import AAEncoder
from utils.equinox.equinox_utils import DistanceDropEdge
from utils.equinox.graph_utils import subgraph


class LocalEncoder(eqx.Module):
    """Local encoder module using Equinox."""

    historical_steps: int
    node_dim: int
    edge_dim: int
    embed_dim: int
    num_heads: int
    dropout: float
    local_radius: float

    drop_edge: DistanceDropEdge
    aa_encoder: AAEncoder

    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_radius: float = 50,
        *,
        key: jax.random.PRNGKey,
    ) -> None:
        """Initialize LocalEncoder.

        Args:
            historical_steps: Number of historical time steps
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            local_radius: Local radius for edge dropping
            parallel: Whether to use parallel computation
            key: PRNG key for initialization
        """
        self.historical_steps = historical_steps
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.local_radius = local_radius

        # Split key for different components
        k1, k2 = jax.random.split(key)

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(
            embed_dim=embed_dim,
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            key=k2,
        )

    # @beartype
    def __call__(
        self, data: dict, *, key: Optional[jax.random.PRNGKey] = None
    ) -> Float[Array, "T N D"]:
        """Forward pass of LocalEncoder.

        Args:
            data: Dictionary containing:
                - x: Node features [N, T, D]
                - positions: Node positions [N, T, 2]
                - edge_index: Edge indices [2, E]
                - padding_mask: Padding mask [N, T]
                - bos_mask: Beginning of sequence mask [N, T]
                - rotate_mat: Rotation matrix [N, 2, 2]
            key: PRNG key for dropout

        Returns:
            Output tensor of shape [T, N, D]
        """
        for t in range(self.historical_steps):
            data[f"edge_index_{t}"], _ = subgraph(
                subset=~data["padding_mask"][:, t], edge_index=data["edge_index"]
            )
            data[f"edge_attr_{t}"] = (
                data["positions"][data[f"edge_index_{t}"][0], t]
                - data["positions"][data[f"edge_index_{t}"][1], t]
            )

        # Process each timestep
        outputs = []
        for t in range(self.historical_steps):

            out_t = self.aa_encoder(
                positions=data["positions"], t=t, bos_mask=data["bos_mask"]
            )  # More efficient to do the calculation inside the hub spoke and actually just pass in positojns

            outputs.append(out_t)

        # Stack outputs along time dimension
        return jnp.stack(outputs)  # [T, N, D]
