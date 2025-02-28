import equinox as eqx
import jax
from jaxtyping import Array, Float, PRNGKeyArray
from beartype import beartype


@beartype
class ReLU(eqx.Module):
    def __call__(self, x: Float[Array, "d=8"], key=None):
        output = jax.nn.relu(x)
        return output

@beartype
class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    relu: ReLU

    def __init__(self, embed_dim: int, dropout_rate: float, keys: PRNGKeyArray):
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[0])
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[1])
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)
        self.relu = ReLU()

    def __call__(
        self,
        nodes: Float[Array, "d=2"],
        key,
    ) -> Float[Array, "d=2"]:

        key1, key2 = jax.random.split(key)
        nodes = self.linear1(nodes)

        nodes = self.relu(nodes)

        nodes = self.dropout1(nodes, key=key1)

        nodes = self.linear2(nodes)
        nodes = self.dropout2(nodes, key=key2)
        return nodes
