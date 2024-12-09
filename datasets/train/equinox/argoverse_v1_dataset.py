from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import jax.numpy as jnp
import equinox as eqx
from pathlib import Path
import pandas as pd
import numpy as np
from processing import process_argoverse

@dataclass
class ArgoverseExample:
    """Single example from Argoverse dataset."""
    x: jnp.ndarray                # [N, 20, 2] Historical trajectories
    positions: jnp.ndarray        # [N, 50, 2] All positions
    edge_index: jnp.ndarray       # [2, N x N - 1] Edge connectivity
    y: Optional[jnp.ndarray]      # [N, 30, 2] Future trajectories (None for test)
    num_nodes: int                # Number of nodes
    padding_mask: jnp.ndarray     # [N, 50] Padding mask
    bos_mask: jnp.ndarray        # [N, 20] Beginning of sequence mask
    rotate_angles: jnp.ndarray    # [N] Rotation angles
    lane_vectors: jnp.ndarray     # [L, 2] Lane vectors
    is_intersections: jnp.ndarray # [L] Intersection indicators
    turn_directions: jnp.ndarray  # [L] Turn directions
    traffic_controls: jnp.ndarray # [L] Traffic controls
    lane_actor_index: jnp.ndarray # [2, E_{A-L}] Lane-actor connections
    lane_actor_vectors: jnp.ndarray # [E_{A-L}, 2] Lane-actor vectors
    seq_id: int                   # Sequence ID
    av_index: int                 # AV index
    agent_index: int              # Agent index
    city: str                     # City name
    origin: jnp.ndarray          # [1, 2] Origin coordinates
    theta: float                  # Rotation angle


class ArgoverseDataset(eqx.Module):
    examples: List[ArgoverseExample]
    split: str
    local_radius: float

    def __init__(self, 
                 root: str,
                 split: str,
                 local_radius: float = 50.0):
        """Initialize Argoverse dataset.
        
        Args:
            root: Path to dataset
            split: One of ['train', 'val', 'test', 'sample']
            local_radius: Radius for local context
        """
        self.split = split
        self.local_radius = local_radius
        
        # Set up paths
        root_path = Path(root)
        if split == 'sample':
            directory = 'forecasting_sample'
        elif split in ['train', 'val', 'test']:
            directory = 'train' if split == 'train' else ('val' if split == 'val' else 'test_obs')
        else:
            raise ValueError(f'Invalid split: {split}')
            
        data_dir = root_path / directory / 'data'
        processed_dir = root_path / directory / 'processed'
        
        # Process or load data
        self.examples = self._load_or_process_data(data_dir, processed_dir)


    def _load_or_process_data(self, 
                            data_dir: Path,
                            processed_dir: Path) -> List[ArgoverseExample]:
        """Load processed data or process raw data."""
        processed_dir.mkdir(parents=True, exist_ok=True)
        examples = []
        
        for raw_path in data_dir.glob('*.csv'):
            processed_path = processed_dir / f'{raw_path.stem}.npz'
            
            if processed_path.exists():
                # Load processed data
                data = np.load(processed_path, allow_pickle=True)
                example = ArgoverseExample(**{k: jnp.array(v) if isinstance(v, np.ndarray) else v 
                                           for k, v in data.items()})
            else:
                # Process raw data
                kwargs = process_argoverse(self.split, str(raw_path), self.local_radius)
                example = ArgoverseExample(**kwargs)
                # Save processed data
                np.savez(processed_path, **{k: v.numpy() if hasattr(v, 'numpy') else v 
                                          for k, v in kwargs.items()})
            
            examples.append(example)
            
        return examples


    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ArgoverseExample:
        return self.examples[idx]