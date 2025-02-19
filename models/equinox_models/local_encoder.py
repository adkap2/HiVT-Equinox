from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray
from beartype import beartype
from .aa_encoder import AAEncoder
from utils.equinox.equinox_utils import DistanceDropEdge
from utils.equinox.graph_utils import subgraph
from .temporal_encoder import TemporalEncoder
from .al_encoder import ALEncoder
from einops import rearrange
class LocalEncoder(eqx.Module):
    """Local encoder module using Equinox."""

    historical_steps: int
    num_temporal_layers: int
    node_dim: int
    edge_dim: int
    embed_dim: int
    num_heads: int
    dropout: float
    local_radius: float

    drop_edge: DistanceDropEdge
    aa_encoder: AAEncoder
    temporal_encoder: TemporalEncoder
    al_encoder: ALEncoder
    def __init__(
        self,
        historical_steps: int,
        node_dim: int,
        edge_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_radius: float = 50,
        num_temporal_layers: int = 4,
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
        self.num_temporal_layers = num_temporal_layers
        # Split key for different components
        k1, k2, k3 = jax.random.split(key, 3)

        self.drop_edge = DistanceDropEdge(local_radius)
        self.aa_encoder = AAEncoder(
            embed_dim=embed_dim,
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            key=k1,
        )
        self.temporal_encoder = TemporalEncoder(historical_steps=historical_steps, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_temporal_layers, dropout=dropout, key=k2)
    
        self.al_encoder = ALEncoder(node_dim=node_dim,edge_dim=edge_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, max_radius=local_radius, key=k3)

    # @beartype
    def __call__(
        self, data: dict, *, key: PRNGKeyArray
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
        # for t in range(self.historical_steps):
            # data[f"edge_index_{t}"], _ = subgraph(
            #     subset=~data["padding_mask"][:, t], edge_index=data["edge_index"]
            # )
            # data[f"edge_attr_{t}"] = (
            #     data["positions"][data[f"edge_index_{t}"][0], t]
            #     - data["positions"][data[f"edge_index_{t}"][1], t]
            # )
            # TODO this is done in the neighbor mask now

        # # Process each timestep
        # outputs = []
        # for t in range(1, self.historical_steps):

        #     out_t = self.aa_encoder(
        #         positions=data["positions"], t=t, bos_mask=data["bos_mask"], padding_mask=data["padding_mask"], key=key
        #     )  # More efficient to do the calculation inside the hub spoke and actually just pass in positojns

        #     outputs.append(out_t)

        key1, key2, key3 = jax.random.split(key, 3)

        def f(t):
            return self.aa_encoder(positions=data["positions"], t=t, bos_mask=data["bos_mask"], padding_mask=data["padding_mask"], key=key1)
        # Vmap

        out = jax.vmap(f)(jnp.arange(1, self.historical_steps+1)) # TODO Confirm that this works
        
        # Process one timestep at a time


        # Stack outputs along time dimension
        # Move dimenison N to front
        out = rearrange(out, "t n d -> n t d")

        # keys = jax.random.split(key2, self.historical_steps)
        # out = jax.vmap(temporal_encoder_f)(out, padding_mask=data["padding_mask"][:, 1: self.historical_steps], key=key)
        def temporal_encoder_f(x, padding_mask, key):
            return self.temporal_encoder(x=x, padding_mask=padding_mask, key=key)

        # Split keys for each element in the batch
        keys = jax.random.split(key2, out.shape[0])  # Split for each batch element
        out = jax.vmap(temporal_encoder_f)(out, 
                                         padding_mask=data["padding_mask"][:, 1: self.historical_steps+1], 
                                         key=keys)
        
        # out = self.al_encoder(temporal_embeddings=out, positions=data["positions"], is_intersections=data["is_intersections"], turn_directions=data["turn_directions"], traffic_controls=data["traffic_controls"], key=key3)

        is_intersection = data["is_intersections"]
        turn_direction = data["turn_directions"]
        traffic_control = data["traffic_controls"]

        lane_vectors = data["lane_vectors"]
        lane_actor_index = data["lane_actor_index"]
        lane_actor_vectors = data["lane_actor_vectors"]


        positions = data["positions"][:, self.historical_steps-1:self.historical_steps+1, :]
        # pos1 = positions[:, self.historical_steps-1:self.historical_steps, :]
        # jax.debug.breakpoint()

        # Or alternatively, if you need to compute from scratch:
        L = lane_vectors.shape[0]  # number of lanes
        lane_start = jnp.zeros((L, 2))  # [L, 2]
        lane_end = lane_vectors  # [L, 2]

        

        def al_encoder_f(temporal_embeddings, key):
            return self.al_encoder(temporal_embeddings=temporal_embeddings, positions=positions, lane_start=lane_start, lane_end=lane_end, is_intersection=is_intersection, turn_direction=turn_direction, traffic_control=traffic_control, key=key)


        keys = jax.random.split(key3, out.shape[0])
        out = jax.vmap(al_encoder_f)(temporal_embeddings=out, key=keys)
        # THIS Should be a vmap
        # breakpoint()
        return out


