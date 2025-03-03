import jax
import torch
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.aa_encoder import AAEncoder as EquinoxAAEncoder
from models.torch_models.aa_encoder import TorchAAEncoder
from models.torch_models.local_encoder import LocalEncoder as TorchLocalEncoder
from models.equinox_models.local_encoder import LocalEncoder as EquinoxLocalEncoder
from models.equinox_models.global_interactor import GlobalInteractor as EquinoxGlobalInteractor
from models.equinox_models.decoder import MLPDecoder as EquinoxMLPDecoder

from utils import TemporalData


# import argoverse

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



def test_aa_encoder():
    # Set random seed
    key: PRNGKeyArray = jax.random.PRNGKey(0)
    torch.manual_seed(0)

    # Test parameters
    historical_steps: int = 20
    node_dim: int = 2
    edge_dim: int = 2
    embed_dim: int = 2
    num_heads: int = 2
    dropout: float = 0.1

    num_nodes: int = 2
    num_temporal_layers: int = 4


    # Create rotation angles (one per actor)
    rotate_angles = torch.tensor([np.pi/4, -np.pi/6])  # example angles: 45° and -30°

    # Create empty rotation matrix
    rotate_mat = torch.empty(num_nodes, 2, 2)  # [2, 2, 2]

    # Calculate sin and cos values
    sin_vals = torch.sin(rotate_angles)  # [2]
    cos_vals = torch.cos(rotate_angles)  # [2]

    # Fill rotation matrix
    rotate_mat[:, 0, 0] = cos_vals    # R_00 = cos(θ)
    rotate_mat[:, 0, 1] = -sin_vals   # R_01 = -sin(θ)
    rotate_mat[:, 1, 0] = sin_vals    # R_10 = sin(θ)
    rotate_mat[:, 1, 1] = cos_vals    # R_11 = cos(θ)



    torch_model = TorchLocalEncoder(
        historical_steps=historical_steps, # int
        node_dim=node_dim, # int
        edge_dim=edge_dim, # int
        embed_dim=embed_dim, # int
        num_heads=num_heads, # int
        dropout=dropout, # float
    ) 

    x_torch: torch.Tensor = torch.zeros(num_nodes, 50, 2, dtype=torch.float) # [N, 50, 2] # Position of the Nodes Local radius = 50

    for t in range(historical_steps):
        x_torch[0, t] = torch.tensor([t * 0.1, t * 0.1])  # diagonal movement

    for t in range(historical_steps):
        theta = t * np.pi / 20  # circular trajectory
        x_torch[1, t] = torch.tensor([np.cos(theta), np.sin(theta)])


    edge_index_torch = torch.tensor([[0, 1],   # Source nodes
                                   [1, 0]])    # Target nodes
    
    # Create positions for 2 nodes over 1 timestep
    positions_torch = x_torch.clone() # Shape: [num_nodes = 2, total_steps = 50, xy = 2]

    y_torch = x_torch[:, historical_steps:] # Shape: [num_nodes = 2, future_steps = 30, xy = 2]
    x_torch = x_torch[:, : historical_steps] # Shape: [num_nodes = 2, historical_steps = 20, xy = 2]
    
    
    # Calculate edge attributes (relative positions)
    edge_attr_torch = positions_torch[0][edge_index_torch[0]] - positions_torch[0][edge_index_torch[1]] # [2, 2]
    

    padding_mask_torch: torch.Tensor = torch.ones(num_nodes, 50, dtype=torch.bool) # [N, 50] # Padding Mask

    
    edge_index_torch = torch.tensor([[0, 1], [1, 0]], dtype=torch.long) # [2, 2]
    # edge_attr_torch = torch.randn(edge_index_torch.shape[1], edge_dim)
    edge_attr_torch = torch.randn(
        edge_index_torch.shape[1], edge_dim
    )  # Shape: [num_edges, embed_dim] # [2, 2]

    bos_mask_torch = torch.zeros((2, historical_steps), dtype=torch.bool)  # [num_nodes, timesteps]
    bos_mask_torch[0, :] = True  # First node is BOS at all timesteps
    # Create identical rotation matrices for both frameworks
    np_rot_mat: np.ndarray = rotate_mat.numpy() # [2, 2, 2]

    # Convert to respective frameworks
    batch_rot_mat_torch: torch.Tensor = torch.tensor(np_rot_mat) # [2, 2, 2]

    is_intersections_torch = torch.zeros((2, historical_steps), dtype=torch.bool) # [2, 20]
    turn_directions_torch = torch.zeros((2, historical_steps), dtype=torch.bool) # [2, 20]
    traffic_controls_torch = torch.zeros((2, historical_steps), dtype=torch.bool) # [2, 20]

    data = {
        'positions': positions_torch,  # Shape: [num_nodes = 2, total_steps = 50, xy = 2]
        'edge_index': edge_index_torch,
        'edge_attr': edge_attr_torch,
        'bos_mask': bos_mask_torch,
        'rotate_mat': batch_rot_mat_torch,
        'padding_mask': padding_mask_torch,
        'x': x_torch,
        'y': y_torch,
    }

    # TODO turn into temporal data object
    torch_output = torch_model(
        data,
    ) # Shape: [batch_size, embed_dim] -> [2, 2]

    eqx_model = EquinoxLocalEncoder(
        historical_steps=historical_steps,
        node_dim=node_dim,
        edge_dim=edge_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout=dropout,
        key=key,
        num_temporal_layers=num_temporal_layers,
    )

    bos_mask_jax: jnp.ndarray = jnp.array(bos_mask_torch.numpy()) # Shape: [batch_size] -> [2]

    positions_jax = jnp.array(positions_torch.numpy()) # [N, 50, 2]


    padding_mask_jax = jnp.array(padding_mask_torch.numpy()) # [N, 50]

    data_jax = {
        'bos_mask': bos_mask_jax,
        'positions': positions_jax,
        'padding_mask': padding_mask_jax,
    }

    eqx_output: jnp.ndarray = eqx_model(
        data_jax,
        key=key,
    ) # Shape: [batch_size, embed_dim] -> [2, 2]

    # Compare outputs
    print(f"[JAX] eqx_output shape: {eqx_output.shape}")
    print(f"[JAX] eqx_output first few values: {eqx_output[0, :5]}")

    print(f"[TORCH] torch_output shape: {torch_output.shape}")
    print(f"[TORCH] torch_output first few values: {torch_output[0, :5]}")


def test_local_encoder_with_argoverse():


        csv_path = "datasets/train/data/1.csv"
        am = ArgoverseMap()
        
        # Print sample data info
        kwargs = process_argoverse(
            split='train',
            raw_path=csv_path,
            am=am,
            radius=50.0
        )

        data = TemporalData(**kwargs)

        # Convert to dictionary
        data_dict = data.to_dict()

        # print(data_dict)

        # Go into each key and convert to jax array might have to convert to numpy first
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = jnp.array(value.numpy())
            elif isinstance(value, np.ndarray):
                data_dict[key] = jnp.array(value)
            else:
                # Leave as is
                pass


        historical_steps = 20
        node_dim = 2
        edge_dim = 2
        embed_dim = 2
        num_heads = 2
        dropout = 0.1
        num_temporal_layers = 4

        key = jax.random.PRNGKey(0)
        eqx_local_encoder = EquinoxLocalEncoder(
            historical_steps=historical_steps,
            node_dim=node_dim,
            edge_dim=edge_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        key=key,
        num_temporal_layers=num_temporal_layers,
    )

        local_encoder_output = eqx_local_encoder(data_dict, key=key)
        

        num_modes = 2


        key = jax.random.PRNGKey(1)
        # TODO add global interactor    
        eqx_global_interactor = EquinoxGlobalInteractor(
            historical_steps=historical_steps,
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_modes=num_modes,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout,
            key=key,
        )

        eqx_global_interactor_output = eqx_global_interactor(data = data_dict, local_embed = local_encoder_output, key=key)


        eqx_decoder = EquinoxMLPDecoder(
            local_channels=embed_dim,
            global_channels=embed_dim,
            future_steps=30,
            num_modes=num_modes,
            key=key,
        )

        out, pi = eqx_decoder(
            local_embed=local_encoder_output,
            global_embed=eqx_global_interactor_output,
            key=key,
        )

        print(f"Equinox decoder output shape: {out.shape}")
        print(f"Equinox decoder output first few values: {out[0, :5]}")

        print(f"Equinox decoder pi shape: {pi.shape}")
        print(f"Equinox decoder pi first few values: {pi[0, :5]}")





if __name__ == "__main__":
    # test_aa_encoder()
    test_local_encoder_with_argoverse()
