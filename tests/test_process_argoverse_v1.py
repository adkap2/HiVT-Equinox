import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import jax
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


def test_csv_exists(sample_csv_path):
    """Test that the CSV file exists and can be read"""
    assert os.path.exists(sample_csv_path), f"CSV file not found at {sample_csv_path}"
    
    # Try to read first few lines to verify file is accessible
    with open(sample_csv_path, 'r') as f:
        first_line = f.readline()
    assert 'TIMESTAMP' in first_line, "CSV header not found"

def test_dataset_creation(dataset):
    """Test basic dataset properties"""
    assert isinstance(dataset, ArgoverseV1Dataset)
    assert dataset._split == "train"
    assert dataset._local_radius == 50.0
    assert len(dataset) > 0


def test_process_argoverse_basic(sample_csv_path, argoverse_map):
    """Test basic processing of an Argoverse sample"""
    
    result = process_argoverse(
        split='train',
        raw_path=sample_csv_path,
        am=argoverse_map,
        radius=50.0
    )
    
    # Check that all expected keys are present
    expected_keys = {
        'x', 'positions', 'edge_index', 'y', 'num_nodes',
        'padding_mask', 'bos_mask', 'rotate_angles',
        'lane_vectors', 'is_intersections', 'turn_directions',
        'traffic_controls', 'lane_actor_index', 'lane_actor_vectors'
    }
    assert set(result.keys()) >= expected_keys


def test_map_features(sample_csv_path, argoverse_map):
    """Test map-related features from Argoverse"""
    
    result = process_argoverse(
        split='train',
        raw_path=sample_csv_path,
        am=argoverse_map,
        radius=50.0
    )
    
    # Test lane features
    assert result['lane_vectors'] is not None
    assert result['is_intersections'] is not None
    assert result['turn_directions'] is not None
    assert result['traffic_controls'] is not None


def test_trajectory_features(sample_csv_path, argoverse_map):
    """Test trajectory-related features"""
    
    result = process_argoverse(
        split='train',
        raw_path=sample_csv_path,
        am=argoverse_map,
        radius=50.0
    )
    
    # Check trajectory shapes
    N = result['num_nodes']  # number of nodes
    assert result['x'].shape[0] == N  # number of nodes
    assert result['x'].shape[2] == 2  # x,y coordinates
    assert result['positions'].shape[0] == N
    assert result['positions'].shape[2] == 2

if __name__ == "__main__":


        csv_path = "datasets/train/data/1.csv"
        am = ArgoverseMap()
        
        # Run tests
        test_csv_exists(csv_path)
        print("✅ CSV test passed!")

        dataset_dir = "datasets"

        train_dir = "datasets/train/data"
        csv_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files")

        # Initialize ArgoMap
        am = ArgoverseMap()
        
        # Test a subset of files (e.g., first 5)
        for csv_file in csv_files[:5]:
            csv_path = os.path.join(train_dir, csv_file)
            print(f"\nTesting file: {csv_file}")
            
        
            test_process_argoverse_basic(csv_path, am)
            print("✅ Basic processing test passed!")

            # jax.debug.breakpoint()

        argo_v1_dataset = ArgoverseV1Dataset(root=dataset_dir, split='sample', transform=None)
    
        argo_v1_dataset.process()

        # test_map_features(csv_path, am)
        # print("✅ Map features test passed!")
        
        test_trajectory_features(csv_path, am)
        print("✅ Trajectory features test passed!")
        
        # Print sample data info
        kwargs = process_argoverse(
            split='train',
            raw_path=csv_path,
            am=am,
            radius=50.0
        )

        data = TemporalData(**kwargs)


        print("\nProcessed data properties:")
        print(f"Number of nodes: {data['num_nodes']}")
        print(f"Historical trajectory shape: {data['x'].shape}")
        print(f"Full trajectory shape: {data['positions'].shape}")
        print(f"Number of lanes: {data['lane_vectors'].shape[0]}")
        print(f"x: {data['x']}")
        print(f"positions: {data['positions']}")
        print(f"edge_index: {data['edge_index']}")
        print(f"y: {data['y']}")
        print(f"num_nodes: {data['num_nodes']}")
        print(f"padding_mask: {data['padding_mask']}")
        print(f"bos_mask: {data['bos_mask']}")
        print(f"rotate_angles: {data['rotate_angles']}")
        print(f"lane_vectors: {data['lane_vectors']}")
        print(f"is_intersections: {data['is_intersections']}")
        print(f"turn_directions: {data['turn_directions']}")
        print(f"traffic_controls: {data['traffic_controls']}")
        print(f"lane_actor_index: {data['lane_actor_index']}")
        print(f"lane_actor_vectors: {data['lane_actor_vectors']}")
        print(f"seq_id: {data['seq_id']}")
        print(f"av_index: {data['av_index']}")
        print(f"agent_index: {data['agent_index']}")
        print(f"city: {data['city']}")
        print(f"origin: {data['origin']}")
        print(f"theta: {data['theta']}")
    