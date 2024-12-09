from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Float

def subgraph(
    subset: Bool[Array, "N"],  # Boolean mask for nodes to keep
    edge_index: Int[Array, "2 E"],  # Edge indices
    edge_attr: Optional[Float[Array, "E D"]] = None,  # Optional edge attributes
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    return_edge_mask: bool = False,
) -> Tuple[Int[Array, "2 E_new"], Optional[Float[Array, "E_new D"]]]:
    """Returns the induced subgraph containing the nodes in subset.
    
    Args:
        subset: Boolean mask indicating which nodes to keep
        edge_index: Edge indices
        edge_attr: Optional edge attributes
        relabel_nodes: If True, relabel nodes to be consecutive
        num_nodes: Total number of nodes (optional)
        return_edge_mask: If True, return the edge mask
        
    Returns:
        new_edge_index: Filtered edge indices
        new_edge_attr: Filtered edge attributes (if provided)
        edge_mask: Boolean mask for edges (if return_edge_mask=True)
    """
    # Create edge mask where both source and target nodes are in subset
    row, col = edge_index[0], edge_index[1]
    edge_mask = subset[row] & subset[col]
    
    # Filter edges
    new_edge_index = edge_index[:, edge_mask]
    
    # Filter edge attributes if they exist
    new_edge_attr = None
    if edge_attr is not None:
        new_edge_attr = edge_attr[edge_mask]
    
    # Relabel nodes if requested
    if relabel_nodes:
        # Create mapping for new node indices
        node_idx = jnp.arange(len(subset))
        mapping = jnp.where(subset, 
                           jnp.cumsum(subset) - 1, 
                           node_idx)  # -1 for unused indices
        
        # Apply mapping to edge indices
        new_edge_index = mapping[new_edge_index]
    
    if return_edge_mask:
        return new_edge_index, new_edge_attr, edge_mask
    else:
        return new_edge_index, new_edge_attr