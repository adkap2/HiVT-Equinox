import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Tuple
from einops import rearrange, repeat
from beartype import beartype


@beartype
class PiMLP(eqx.Module):
    linear1: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm
    linear3: eqx.nn.Linear
    
    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 3)
        self.linear1 = eqx.nn.Linear(input_size, hidden_size, key=keys[0])
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.linear2 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.norm2 = eqx.nn.LayerNorm(hidden_size)
        self.linear3 = eqx.nn.Linear(hidden_size, 1, key=keys[2])
    
    def __call__(self, x: Float[Array, "d"],
                  *,
                    key: PRNGKeyArray = None
                    ) -> Float[Array, "1"]:
        x = self.linear1(x)
        x = self.norm1(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = jax.nn.relu(x)
        x = self.linear3(x)
        return x


@beartype
class AggregationMLP(eqx.Module):
    linear: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    
    def __init__(self, input_size: int, hidden_size: int, *, key: PRNGKeyArray):
        self.linear = eqx.nn.Linear(input_size, hidden_size, key=key)
        self.norm = eqx.nn.LayerNorm(hidden_size)
    
    def __call__(self, x: Float[Array, "d"], *, key: PRNGKeyArray = None) -> Float[Array, "..."]:
        x = self.linear(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        return x


@beartype
class LocationMLP(eqx.Module):
    linear1: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear
    
    def __init__(self, hidden_size: int, future_steps: int, *, key: PRNGKeyArray):
        keys = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(hidden_size, hidden_size, key=keys[0])
        self.norm = eqx.nn.LayerNorm(hidden_size)
        self.linear2 = eqx.nn.Linear(hidden_size, future_steps * 2, key=keys[1])
    
    def __call__(self, x: Float[Array, "d"], *, key: PRNGKeyArray = None) -> Float[Array, "..."]:
        x = self.linear1(x)
        x = self.norm(x)
        x = jax.nn.relu(x)
        x = self.linear2(x)
        return x



class MLPDecoder(eqx.Module):

    aggr_embed: AggregationMLP
    loc: LocationMLP
    input_size: int
    hidden_size: int
    future_steps: int
    num_modes: int
    multihead_proj: eqx.Module
    pi_mlp: PiMLP

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 *, key: jax.random.PRNGKey):
        
        keys = jax.random.split(key, 4)
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes

        # Add multihead projection at decoder start
        self.multihead_proj = eqx.nn.Linear(
            global_channels, num_modes * global_channels, key=keys[0]
        )


        # Aggregation MLP
        self.aggr_embed = AggregationMLP(input_size=self.input_size + self.hidden_size, hidden_size=self.hidden_size, key=keys[0])

        # Location MLP
        self.loc = LocationMLP(hidden_size=self.hidden_size, future_steps=self.future_steps, key=keys[1])

        # TODO skipping uncertainty for now

        # Pi MLP
        self.pi_mlp = PiMLP(input_size=self.input_size + self.hidden_size, hidden_size=self.hidden_size, key=keys[3])


    def __call__(self,
                 local_embed: Float[Array, "N d"],
                 global_embed: Float[Array, "N d"],
                 *, key: jax.random.PRNGKey) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        
        global_embed = jax.vmap(self.multihead_proj)(global_embed)
        # Combine reshape and transpose in one clean operation
        global_embed = rearrange(
            global_embed,
            'n (m d) -> m n d',
            m=self.num_modes,
            d=self.input_size
        )  # [M, N, d]
        # Calculate pi
        # Expand local_embed to match global_embed's modes
        local_expanded = repeat(
            local_embed,
            'n d -> m n d',
            m=self.num_modes
        )  # [M, N, d] 

        pi_input = jnp.concatenate([local_expanded, global_embed], axis=-1)
        # Input shape is [M, N, D+D] (2, 9, 4)
        pi = jax.vmap(jax.vmap(self.pi_mlp))(pi_input)

        aggr_input = jnp.concatenate([global_embed, local_expanded], axis=-1)
        out = jax.vmap(jax.vmap(self.aggr_embed))(aggr_input)

        # Location
        loc = jax.vmap(jax.vmap(self.loc))(out)
        # Convert from [M, N, H*2] to [M, N, H, 2]
        loc = rearrange(
            loc,
            'm n (h d) -> m n h d',
            h=self.future_steps,
            d=2
        )  # [M, N, H, 2]

        # TODO skipping uncertainty for now
        out = loc

        return out, pi
