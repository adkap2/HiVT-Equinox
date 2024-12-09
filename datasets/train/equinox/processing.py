import jax.numpy as jnp
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from itertools import permutations, product

def process_argoverse(split: str,
                     raw_path: str,
                     radius: float) -> Dict:
    """Process raw Argoverse data file."""
    df = pd.read_csv(raw_path)

    # Get timestamps
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[:20]

    # Filter actors
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)


    # Get AV and agent info
    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])
    city = df['CITY_NAME'].values[0]


    # Calculate scene center and rotation
    origin = jnp.array([av_df[19]['X'], av_df[19]['Y']], dtype=jnp.float32)
    av_heading_vector = origin - jnp.array([av_df[18]['X'], av_df[18]['Y']], dtype=jnp.float32)
    theta = jnp.arctan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                            [jnp.sin(theta), jnp.cos(theta)]], dtype=jnp.float32)
    

    # Initialize arrays
    x = jnp.zeros((num_nodes, 50, 2), dtype=jnp.float32)
    edge_index = jnp.array(list(permutations(range(num_nodes), 2))).T
    padding_mask = jnp.ones((num_nodes, 50), dtype=bool)
    bos_mask = jnp.zeros((num_nodes, 20), dtype=bool)
    rotate_angles = jnp.zeros(num_nodes, dtype=jnp.float32)


    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        
        # Update padding mask
        padding_mask = padding_mask.at[node_idx, node_steps].set(False)
        if padding_mask[node_idx, 19]:  # make no predictions for actors unseen at current time step
            padding_mask = padding_mask.at[node_idx, 20:].set(True)
        
        # Process positions
        xy = jnp.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)
        x = x.at[node_idx, node_steps].set(jnp.matmul(xy - origin, rotate_mat))
        
        # Process historical steps
        node_historical_steps = [step for step in node_steps if step < 20]
        
        # Calculate heading for each actor
        if len(node_historical_steps) > 1:
            # Calculate heading vector from last two positions
            heading_vector = (x[node_idx, node_historical_steps[-1]] - 
                            x[node_idx, node_historical_steps[-2]])
            rotate_angles = rotate_angles.at[node_idx].set(
                jnp.arctan2(heading_vector[1], heading_vector[0])
            )
        else:
            # Not enough historical steps - mask future predictions
            padding_mask = padding_mask.at[node_idx, 20:].set(True)


    # Create beginning of sequence mask
    bos_mask = bos_mask.at[:, 0].set(~padding_mask[:, 0])
    bos_mask = bos_mask.at[:, 1:20].set(
        padding_mask[:, :19] & ~padding_mask[:, 1:20]
    )


    # Process positions and displacements
    positions = x.copy()
    
    # Future displacements (t >= 20)
    future_mask = (padding_mask[:, 19].reshape(-1, 1) | 
                  padding_mask[:, 20:]).reshape(-1, 1, 1)
    x = x.at[:, 20:].set(
        jnp.where(future_mask,
                  jnp.zeros((num_nodes, 30, 2)),
                  x[:, 20:] - x[:, 19:20])
    )