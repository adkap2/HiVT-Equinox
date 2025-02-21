import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import numpy as np
from einops import rearrange, repeat
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.equinox_models.local_encoder import LocalEncoder as EquinoxLocalEncoder
from models.equinox_models.global_interactor import GlobalInteractor as EquinoxGlobalInteractor
from models.equinox_models.decoder import MLPDecoder as EquinoxMLPDecoder
from utils import TemporalData

from jaxtyping import Array, Float, PRNGKeyArray
from jax.random import PRNGKey

# Go up two levels (from tests/ to parent directory) and then to argoverse-api
argoverse_path = os.path.join(
    os.path.dirname(  # up from tests/
        os.path.dirname(  # up from HiVT-Equinox/
            os.path.dirname(os.path.abspath(__file__))  # current file
        )
    ),
    "argoverse-api"
)
sys.path.append(argoverse_path)

from argoverse.map_representation.map_api import ArgoverseMap
from datasets.torch.argoverse_v1_dataset import process_argoverse, ArgoverseV1Dataset


class HiVTTest:
    def __init__(self,
                 historical_steps: int = 20,
                 future_steps: int = 30,
                 num_modes: int = 6,
                 node_dim: int = 2,
                 edge_dim: int = 2,
                 embed_dim: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 num_temporal_layers: int = 4,
                 num_global_layers: int = 3,
                 local_radius: float = 50,
                 *,
                 key: PRNGKeyArray):

        keys = jax.random.split(key, 4)
        
        # Initialize components
        self.local_encoder = EquinoxLocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            local_radius=local_radius,
            key=keys[0],
            num_temporal_layers=num_temporal_layers,
        )

        self.global_interactor = EquinoxGlobalInteractor(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
            key=keys[1],
        )

        self.decoder = EquinoxMLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=future_steps,
            num_modes=num_modes,
            key=keys[2]
        )

    def __call__(self, data: dict, *, key: jax.random.PRNGKey):
        # Forward pass through entire model
        local_embed = self.local_encoder(data, key=key)
        global_embed = self.global_interactor(data=data, local_embed=local_embed, key=key)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed, key=key)
        return y_hat, pi

def test_hivt_with_argoverse():
    # Set random seed
    key = PRNGKey(0)
    torch.manual_seed(0)

    # Load and process Argoverse data
    csv_path = "datasets/train/data/1.csv"
    am = ArgoverseMap()
    kwargs = process_argoverse(
        split='train',
        raw_path=csv_path,
        am=am,
        radius=50.0
    )
    data = TemporalData(**kwargs)
    data_dict = data.to_dict()

    # Convert torch tensors to jax arrays
    for k, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[k] = jnp.array(value.numpy())
        elif isinstance(value, np.ndarray):
            data_dict[k] = jnp.array(value)
        else:
            # Leave as is
            pass

    # # Initialize model
    model = HiVTTest(
        historical_steps=20,
        future_steps=30,
        num_modes=2,
        node_dim=2,
        edge_dim=2,
        embed_dim=2,
        num_heads=2,
        dropout=0.1,
        num_temporal_layers=4,
        num_global_layers=2,
        local_radius=50.0,
        key=key
    )

    # Forward pass
    y_hat, pi = model(data_dict, key=key)

    # Print shapes and sample values
    print("\nModel Outputs:")
    print(f"Trajectory predictions shape: {y_hat.shape}")  # Should be [F, N, H, 4]
    print(f"Mode probabilities shape: {pi.shape}")        # Should be [N, F]
    
    print("\nSample values:")
    print(f"First trajectory prediction: {y_hat[0,0,0]}")
    print(f"First mode probability: {pi[0]}")

if __name__ == "__main__":
    test_hivt_with_argoverse()