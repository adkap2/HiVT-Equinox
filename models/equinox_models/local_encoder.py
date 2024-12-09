from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int

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
    
    def __init__(self,
                 historical_steps: int,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 local_radius: float = 50,
                 *,
                 key: jax.random.PRNGKey) -> None:
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
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            key=k2
        )

    def __call__(self,
                 data: dict,
                 *,
                 key: Optional[jax.random.PRNGKey] = None) -> Float[Array, "T N D"]:
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
            data[f'edge_index_{t}'], _ = subgraph(
                subset=~data['padding_mask'][:, t],
                edge_index=data['edge_index']
            )
            data[f'edge_attr_{t}'] = (data['positions'][data[f'edge_index_{t}'][0], t] - 
                        data['positions'][data[f'edge_index_{t}'][1], t])

        # Process each timestep
        outputs = []
        for t in range(self.historical_steps):
            # Generate key for this timestep
            step_key = None if key is None else jax.random.fold_in(key, jnp.array(t))
            

            
            # Apply edge dropping
            edge_index_t, edge_attr_t = self.drop_edge(
                data[f'edge_index_{t}'], data[f'edge_attr_{t}'], key=step_key
            )

            # print("edge_index_t.shape", edge_index_t.shape)
            # print("edge_attr_t.shape", edge_attr_t.shape)
            # print("nodes.shape", jnp.asarray(data['x'][:, t]).shape)
            # print("rotate_mat.shape", jnp.asarray(data['rotate_mat']).shape)
            # print("bos_mask.shape", jnp.asarray(data['bos_mask'][:, t]).shape)
            # print("edge_index_t.shape", edge_index_t.shape)
            #TODO add key here
            # Process through AA encoder
            out_t = self.aa_encoder(
                nodes=jnp.asarray(data['x'][:, t]), # Explicitly convert to jnp array
                t=t,
                edge_index=jnp.asarray(edge_index_t),
                edge_attr=jnp.asarray(edge_attr_t),
                bos_mask=jnp.asarray(data['bos_mask'][:, t]),
                rotate_mat=jnp.asarray(data['rotate_mat']),
                # key=step_key
            )

            outputs.append(out_t)
        
        # Stack outputs along time dimension
        return jnp.stack(outputs)  # [T, N, D]
