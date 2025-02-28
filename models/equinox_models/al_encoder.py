import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Bool
from models.equinox_models.embedding import SingleInputEmbedding

from einops import rearrange, repeat
from beartype import beartype
from models.equinox_models.mlp import MLP


@beartype
class ALEncoder(eqx.Module):

    _lane_embed: eqx.nn.Embedding
    _is_intersection_embed: eqx.nn.Embedding
    _turn_direction_embed: eqx.nn.Embedding
    _traffic_control_embed: eqx.nn.Embedding

    mlp: MLP
    attention: eqx.nn.MultiheadAttention
    lin_self: eqx.nn.Linear
    attn_dropout: float

    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    embed_dim: int
    num_heads: int
    dropout: float
    max_radius: float

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        max_radius: float,
        *,
        key: PRNGKeyArray
    ):

        keys = jax.random.split(key, 10)

        self._lane_embed = SingleInputEmbedding(
            in_channel=7, out_channel=embed_dim, key=keys[0]
        )  # TODO currently hardcoded to 7 features
        self._is_intersection_embed = (
            jax.random.normal(key, (2, embed_dim)) * 0.02
        )  # matches PyTorch's std=.02
        self._turn_direction_embed = (
            jax.random.normal(keys[1], (3, embed_dim)) * 0.02
        )  # matches PyTorch's std=.02
        self._traffic_control_embed = (
            jax.random.normal(keys[2], (2, embed_dim)) * 0.02
        )  # matches PyTorch's std=.02

        self.mlp = MLP(embed_dim, dropout, keys[3:5])
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads, query_size=embed_dim, key=keys[6]
        )
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[7])
        self.attn_dropout = dropout

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[8])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[9])

        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_radius = max_radius

    def __call__(
        self,
        temporal_embedding: Float[Array, "d"],  # [embed_dim]
        pos: Float[Array, "t=2 xy=2"],
        lane_start: Float[Array, "L xy=2"],
        lane_end: Float[Array, "L xy=2"],
        is_intersection: Array,  # [L]
        turn_direction: Array,  # [L]
        traffic_control: Array,  # [L]
        *,
        key: PRNGKeyArray
    ) -> Float[Array, "d"]:

        # Hub Spoke

        rotate_mat = self.create_rotate_mat(pos[-1] - pos[0])

        agent_lane_rel_pos = lane_start - pos[-1]

        lane_vecs = lane_end - lane_start

        neighbor_mask = self.create_neighbor_mask(agent_lane_rel_pos)

        rotated_lane_vecs = jax.vmap(lambda lane_vec: rotate_mat @ lane_vec)(lane_vecs)
        rotated_agent_lane_rel_pos = jax.vmap(
            lambda agent_lane_rel_pos: rotate_mat @ agent_lane_rel_pos
        )(agent_lane_rel_pos)

        # # For [L, 2] shape:
        is_intersection = repeat(is_intersection, "L -> L 1")
        turn_direction = repeat(turn_direction, "L -> L 1")
        traffic_control = repeat(traffic_control, "L -> L 1")

        # Concatenate the features
        features = jnp.hstack(
            [
                rotated_lane_vecs,
                rotated_agent_lane_rel_pos,
                is_intersection,
                turn_direction,
                traffic_control,
            ]  # [L, 10]
        )

        lane_features = jax.vmap(self._lane_embed)(features)

        # Apply attention
        query = rearrange(temporal_embedding, "d -> 1 d")

        neighbor_mask = rearrange(neighbor_mask, "L -> 1 L")
        mha = self.attention(
            query=query, key_=lane_features, value=lane_features, mask=neighbor_mask
        )

        new_msg = rearrange(mha, "1 d -> d")

        gate = jax.nn.sigmoid(self.lin_ih(new_msg) + self.lin_hh(temporal_embedding))

        output = new_msg + gate * (self.lin_self(temporal_embedding) - new_msg)

        output = output + self.mlp(self.norm2(output), key=key)

        return output

    def create_rotate_mat(self, dpos: Float[Array, "xy=2"]) -> Float[Array, "xy=2 xy=2"]:
        """Create rotation matrix from displacement vector.

        Args:
            dpos: Displacement vector [2] representing direction of motion

        Returns:
            rotate_mat: 2x2 rotation matrix that aligns with direction of motion
        """
        # Compute rotation angle from displacement vector
        rotate_angle = jnp.arctan2(dpos[1], dpos[0])  # scalar

        # Compute sin and cos values
        sin_val = jnp.sin(rotate_angle)  # scalar
        cos_val = jnp.cos(rotate_angle)  # scalar

        # Create rotation matrix using jnp.array instead of zeros + at
        rotate_mat = jnp.array([[cos_val, -sin_val], [sin_val, cos_val]])

        return rotate_mat

    def create_neighbor_mask(
        self, agent_lane_rel_pos: Float[Array, "L xy=2"]
    ) -> Bool[Array, "L"]:
        """Create neighbor mask based on agent's lane relative position.

        Args:
            agent_lane_rel_pos: Relative position of agent to lane [2]

        Returns:
            neighbor_mask: Boolean mask indicating if agent is in the lane
        """
        # return jnp.linalg.norm(agent_lane_rel_pos) < self.max_radius  # Example threshold
        # Calculate norm along last axis (xy dimension) for each lane
        distances = jnp.linalg.norm(agent_lane_rel_pos, axis=1)  # [L]

        # Create mask where True means "keep this lane"
        neighbor_mask = distances < self.max_radius  # [L]

        return neighbor_mask
