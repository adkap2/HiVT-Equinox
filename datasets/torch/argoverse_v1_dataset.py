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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData

from einops import rearrange, reduce, repeat, einsum
import jax

from jaxtyping import Array, Float, Int, Scalar, Bool, PRNGKeyArray

class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        print("self._processed_paths", self._processed_paths)

        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        for raw_path in tqdm(self.raw_paths):
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            # Pass in the kwargs to create the temporal data from the dictionary of data
            data = TemporalData(**kwargs) # This is where temporal data is created
            jax.debug.breakpoint()
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df: pd.DataFrame = pd.read_csv(raw_path) # Read the CSV file into a pandas DataFrame
    # print("df", df.head())
    # breakpoint()

    # filter out actors that are unseen during the historical time steps
    timestamps: List[int] = list(np.sort(df['TIMESTAMP'].unique())) # [Time_Steps] # There are multiple datapoints for the same timestamp as each one represents a different object/vehicle
    # print("timestamps", timestamps)
    # breakpoint()
    historical_timestamps: List[int] = timestamps[: 20] # [Historical_T] # Everything up to index 20. SO evertthing before 20 is in the past 
    # print("historical_timestamps", historical_timestamps)
    # breakpoint()
    historical_df: pd.DataFrame = df[df['TIMESTAMP'].isin(historical_timestamps)] # [Historical_N, Historical_T] # Historical Dataframe
    # print("historical_df", historical_df.head())
    # breakpoint()
    actor_ids: List[int] = list(historical_df['TRACK_ID'].unique()) # [N] # Unique actor IDs # This could be other vehicles
    # print("actor_ids", actor_ids)
    # breakpoint()
    df: pd.DataFrame = df[df['TRACK_ID'].isin(actor_ids)] # [N, T] # Dataframe # Number of Nodes Time Steps 
    num_nodes: int = len(actor_ids)

    av_df  = df[df['OBJECT_TYPE'] == 'AV'].iloc # av_df <pandas.core.indexing._iLocIndexer object at 0x314513860> # Focus on only AV object
    # print("av_df", av_df)
    # breakpoint()
    av_index: int = actor_ids.index(av_df[0]['TRACK_ID']) # Index of the AV in the actor_ids list
    # print("av_index", av_index)
    # breakpoint()
    agent_df: pd.DataFrame = df[df['OBJECT_TYPE'] == 'AGENT'].iloc # agent_df <pandas.core.indexing._iLocIndexer object at 0x314513860>
    # print("agent_df", agent_df)
    # breakpoint()
    agent_index: int = actor_ids.index(agent_df[0]['TRACK_ID']) # Index of the Agent in the actor_ids list 
    # print("agent_index", agent_index)
    # breakpoint()
    city: str = df['CITY_NAME'].values[0] # City Name

    # make the scene centered at AV
    origin: torch.Tensor  = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float) # [2] # Origin of the Scene # We start at timestep 20 which we will get xy coordinates as origin
    # print("origin", origin)
    # breakpoint()
    av_heading_vector: torch.Tensor = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float) # [2] # Heading Vector of the AV Take the difference in trajectory from last xy to current xy at the 20th timestep
    # print("av_heading_vector", av_heading_vector)
    # breakpoint()
    theta: torch.Tensor = torch.atan2(av_heading_vector[1], av_heading_vector[0]) # [1] # Rotation Angle of the Scene # Get the angle 
    # print("theta", theta)
    # breakpoint()
    rotate_mat: torch.Tensor  = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]]) # [2, 2] # Rotation Matrix # This is the rotation matrix that all neighbors need to be rotated to to be aligned with coordinate from of AV Vehicle
    # print("rotate_mat", rotate_mat)
    # breakpoint()

    # initialization
    x: torch.Tensor = torch.zeros(num_nodes, 50, 2,  dtype=torch.float) # [N, 50, 2] # Position of the Nodes Looking at total of 50 timesteps, 20 prior and 30 ahead and xy dim=2
    # print("x", x) 
    # breakpoint()
    # edge_index: torch.Tensor = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous() # [2, E] # Edge Index where E is the number of edges
    
    # Create all pairs of nodes using einops
    source_nodes = repeat(
        torch.arange(num_nodes),
        'n -> (n t)',              # repeat each source node for each target
        t=num_nodes
    )

    target_nodes = repeat(
        torch.arange(num_nodes),
        't -> (n t)',              # repeat all target nodes for each source
        n=num_nodes
    )

    # Stack and filter self-loops
    edge_index = rearrange(
        [source_nodes, target_nodes],
        'st (n t) -> st (n t)',    # st=source/target dimension (2), n=num_nodes, t=num_nodes
        n=num_nodes
    )

    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask].contiguous()

    
    
    # print("edge_index", edge_index)
    # breakpoint()
    padding_mask: torch.Tensor = torch.ones(num_nodes, 50, dtype=torch.bool) # [N, 50] # Padding Mask
    # print("padding_mask", padding_mask)
    # breakpoint()
    bos_mask: torch.Tensor = torch.zeros(num_nodes, 20, dtype=torch.bool) # [N, 20] # Beginning of Sequence Mask # First 20 timesteps 
    rotate_angles: torch.Tensor = torch.zeros(num_nodes, dtype=torch.float) # [N] # Rotation Angles of the Nodes

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        # xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        
        # Stack coordinates using einops
        xy = rearrange(
            torch.tensor([actor_df['X'].values, actor_df['Y'].values]),
            'coords points -> points coords',
            coords=2
        ).float()

        
        # x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat) # Matrix Multiplication of the Position of the Nodes and the Rotation Matrix
        # node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps)) #   I think this filters out the node steps less than 20 and only looks at the ones greater or equal to 20
        
        # Rotate coordinates using einsum
        x[node_idx, node_steps] = einsum(
            xy - origin,           # [T, 2]
            rotate_mat,           # [2, 2]
            'time d, d r -> time r'  # time=timesteps, d=input dim, r=rotated dim
        )
        
        # Filter historical steps (could use einops but simple list comp is clearer here)
        node_historical_steps = [step for step in node_steps if step < 20]

        # Calculate the Rotation Angle of the Nodes
        # At every time step, the rotation angle is the angle between the heading vector of the nodes from the last two time steps
        
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]] # Heading Vector of the Nodes from the last two time steps
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0]) # Rotation Angle of the Nodes
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True # Just put padding in that location and ignore it

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1:20] = padding_mask[:, :19] & ~padding_mask[:, 1:20] # Its not really a beginning of sequence mask, it tells us if the agent is in the current scene but not in the previous scene

    positions = x.clone()
    # x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
    #                         torch.zeros(num_nodes, 30, 2),
    #                         x[:, 20:] - x[:, 19].unsqueeze(-2)) # TODO [N, 30, 2] # Replace with EINOPS
    # x[:, 1:20] = torch.where((padding_mask[:, :19] | padding_mask[:, 1:20]).unsqueeze(-1),
    #                           torch.zeros(num_nodes, 19, 2), # TODO this calculation should be done in the local endoder. Basically if you don't have a previous position, you can't calculate the difference
    #                           x[:, 1:20] - x[:, :19]) # [N, 19, 2] # TODO Replace with EINOPS 
    # x[:, 0] = torch.zeros(num_nodes, 2) # [N, 2] # Position of the Nodes at Time Step 0


    # Future positions (t >= 20)
    mask_future = repeat(
        padding_mask[:, 19:20] | padding_mask[:, 20:],  # Make sure first part is [N, 1] to broadcast
        'n t -> n t xy',
        xy=2
    )
    x[:, 20:] = torch.where(
        mask_future,
        torch.zeros(num_nodes, 30, 2),
        x[:, 20:] - repeat(x[:, 19:20], 'n t xy -> n (t r) xy', r=30)  # Expand properly
    )

    # Historical positions (1 <= t < 20)
    mask_historical = repeat(
        padding_mask[:, :19] | padding_mask[:, 1:20],
        'n t -> n t xy',
        xy=2
    )
    x[:, 1:20] = torch.where(
        mask_historical,
        torch.zeros(num_nodes, 19, 2),
        x[:, 1:20] - x[:, :19]
    )

    # Initial position (t = 0)
    x[:, 0] = torch.zeros(num_nodes, 2)


    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    # node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()

    # Stack coordinates using einops
    node_positions_19 = rearrange(
        torch.tensor([df_19['X'].values, df_19['Y'].values]),
        'coords points -> points coords',
        coords=2
    ).float()


    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius) # Get the Lane Features

    y = None if split == 'test' else x[:, 20:] # Future Trajectories of the Nodes
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]



    return {
        'x': x[:, :20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': repeat(origin, 'xy -> 1 xy'),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: Float[Array, "N 2"],
                      origin: Float[Array, "2"],
                      rotate_mat: Float[Array, "2 2"],
                      city: str,
                      radius: float) -> Tuple[
                Float[Array, "L 2"],      # lane_vectors
                Float[Array, "L"],        # is_intersections
                Float[Array, "L"],        # turn_directions
                Float[Array, "L"],        # traffic_controls
                Float[Array, "2 E"],      # lane_actor_index
                Float[Array, "E 2"]       # lane_actor_vectors
            ]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    # node_positions0 = torch.matmul(node_positions - origin, rotate_mat).float() # rotation matrix is used to rotate the node positions to the AV's frame of reference

    node_positions = einsum(
    node_positions - origin,  # [N, 2]
    rotate_mat,              # [2, 2]
    'n d, d r -> n r'       # n=nodes, d=input dim, r=rotated dim
    ).float()


    for lane_id in lane_ids:
        # lane_centerline0 = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        # With einops
        # Fixed version
        lane_centerline = rearrange(
            torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)),  # [L, 3]
            'l c -> l c',  # Keep original shape
        ).float()[:, :2]  # Then slice first two columns`


        # lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        # With einops
        lane_centerline = einsum(
            lane_centerline - origin,  # [L, 2]
            rotate_mat,               # [2, 2]
            'l d, d r -> l r'        # l=lane points, d=input dim, r=rotated dim
        )

        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        # is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))

        # # With einops
        is_intersections.append(
            repeat(
                torch.tensor(is_intersection, dtype=torch.uint8),  # single value
                '-> c',                                           # expand to count positions
                c=count
            )
        )


        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        # turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        
        # traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))

        # With einops
        turn_directions.append(
            repeat(
                torch.tensor(turn_direction, dtype=torch.uint8),  # single value
                '-> c',                                          # expand to count positions
                c=count
            )
        )

        traffic_controls.append(
            repeat(
                torch.tensor(traffic_control, dtype=torch.uint8),  # single value
                '-> c',                                           # expand to count positions
                c=count
            )
        )

    # lane_positions = torch.cat(lane_positions, dim=0)
    # lane_vectors = torch.cat(lane_vectors, dim=0)
    # is_intersections = torch.cat(is_intersections, dim=0)
    # turn_directions = torch.cat(turn_directions, dim=0)
    # traffic_controls = torch.cat(traffic_controls, dim=0)

    # With einops
    lane_positions = rearrange(lane_positions, 'list l xy -> (list l) xy')
    lane_vectors = rearrange(lane_vectors, 'list l xy -> (list l) xy')
    is_intersections = rearrange(is_intersections, 'list l -> (list l)')
    turn_directions = rearrange(turn_directions, 'list l -> (list l)')
    traffic_controls = rearrange(traffic_controls, 'list l -> (list l)')


    # lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()

    # With einops
    lane_indices = torch.arange(lane_vectors.size(0))
    node_indices = torch.tensor(node_inds)

    lane_actor_index = rearrange(
        torch.cartesian_prod(lane_indices, node_indices),  # [P, 2] where P = L * N
        'p two -> two p'                                   # transpose to [2, P]
    ).contiguous()


    # lane_actor_vectors = \
    #     lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    
    # With einops
    lane_actor_vectors = (
        repeat(
            lane_positions,                # [L, 2]
            'l xy -> (l n) xy',           # Repeat each lane position for each node
            n=len(node_inds)              # n = number of nodes
        ) - 
        repeat(
            node_positions,               # [N, 2]
            'n xy -> (l n) xy',          # Repeat all node positions for each lane
            l=lane_vectors.size(0)        # l = number of lanes
        )
    )

    # mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius

    # With einops
    mask = (reduce(
        lane_actor_vectors ** 2,
        'pairs coords -> pairs',          # More descriptive names
        reduction='sum'
    ).sqrt() < radius)

    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors


# DO this as a data class