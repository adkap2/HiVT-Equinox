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
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Bool

@dataclass
class TemporalData:
    """Temporal data container for trajectory data."""
    
    x: Optional[Float[Array, "N T D"]] = None                    # Node features
    positions: Optional[Float[Array, "N T 2"]] = None           # Node positions
    edge_index: Optional[Int[Array, "2 E"]] = None             # Edge connectivity
    edge_attrs: Optional[List[Float[Array, "E 2"]]] = None     # Edge attributes per timestep
    y: Optional[Float[Array, "N T 2"]] = None                  # Target/labels
    num_nodes: Optional[int] = None                            # Number of nodes
    padding_mask: Optional[Bool[Array, "N T"]] = None          # Padding mask
    bos_mask: Optional[Bool[Array, "N T"]] = None             # Beginning of sequence mask
    rotate_angles: Optional[Float[Array, "N"]] = None          # Rotation angles
    lane_vectors: Optional[Float[Array, "L 2"]] = None         # Lane vectors
    is_intersections: Optional[Bool[Array, "L"]] = None        # Intersection indicators
    turn_directions: Optional[Int[Array, "L"]] = None          # Turn directions
    traffic_controls: Optional[Bool[Array, "L"]] = None        # Traffic controls
    lane_actor_index: Optional[Int[Array, "2 E_AL"]] = None    # Lane-actor connections
    lane_actor_vectors: Optional[Float[Array, "E_AL 2"]] = None # Lane-actor vectors
    seq_id: Optional[int] = None                               # Sequence ID
    
    def __post_init__(self):
        """Process edge attributes if provided."""
        if self.edge_attrs is not None and self.x is not None:
            self.edge_attrs_dict = {
                f'edge_attr_{t}': self.edge_attrs[t] 
                for t in range(self.x.shape[1])
            }

class DistanceDropEdge(eqx.Module):
    """Drop edges based on distance threshold."""
    
    max_distance: Optional[float]
    
    def __init__(self, max_distance: Optional[float] = None):
        self.max_distance = max_distance
    
    def __call__(self,
                 edge_index: Int[Array, "2 E"],
                 edge_attr: Float[Array, "E 2"],
                 *,
                 key: Optional[jax.random.PRNGKey] = None  # Added for JAX compatibility
                 ) -> Tuple[Int[Array, "2 E_new"], Float[Array, "E_new 2"]]:
        """Filter edges based on distance threshold.
        
        Args:
            edge_index: Edge indices [2, E]
            edge_attr: Edge attributes [E, 2]
            key: PRNG key (unused, but kept for API consistency)
            
        Returns:
            Filtered edge_index and edge_attr
        """
        if self.max_distance is None:
            return edge_index, edge_attr
            
        # Unpack edge indices
        row, col = edge_index[0], edge_index[1]
        
        # Calculate distances and create mask
        mask = jnp.linalg.norm(edge_attr, axis=-1) < self.max_distance
        
        # Filter edges
        new_edge_index = jnp.stack([
            row[mask],
            col[mask]
        ], axis=0)
        
        new_edge_attr = edge_attr[mask]
        
        return new_edge_index, new_edge_attr

# def init_linear(key: jax.random.PRNGKey, 
#                 shape: Tuple[int, ...], 
#                 dtype: jnp.dtype = jnp.float32) -> Array:
#     """Initialize linear layer weights using Xavier uniform."""
#     bound = jnp.sqrt(6.0 / sum(shape))
#     return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)

# def init_embedding(key: jax.random.PRNGKey, 
#                    shape: Tuple[int, ...], 
#                    dtype: jnp.dtype = jnp.float32) -> Array:
#     """Initialize embedding weights using normal distribution."""
#     return jax.random.normal(key, shape=shape, dtype=dtype) * 0.02

# def init_layer_norm(shape: Tuple[int, ...], 
#                     dtype: jnp.dtype = jnp.float32) -> Tuple[Array, Array]:
#     """Initialize LayerNorm parameters."""
#     weight = jnp.ones(shape, dtype=dtype)
#     bias = jnp.zeros(shape, dtype=dtype)
#     return weight, bias