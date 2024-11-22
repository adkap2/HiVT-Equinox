import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding


class AAEncoder(eqx.Module):
    _center_embed: SingleInputEmbedding
    _nbr_embed: MultipleInputEmbedding
    lin_q: eqx.nn.Linear
    lin_k: eqx.nn.Linear
    lin_v: eqx.nn.Linear
    lin_self: eqx.nn.Linear
    attn_dropout: float
    lin_ih: eqx.nn.Linear
    lin_hh: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    proj_drop: float
    # Layer Norms
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    # Historical steps
    historical_steps: int
    embed_dim: int
    num_heads: int
    dropout: float
    mlp: eqx.nn.Sequential
    bos_token: jnp.ndarray
    def __init__(self, historical_steps, node_dim, edge_dim, embed_dim, num_heads=8, dropout=0.1, *, key: Optional[PRNGKeyArray] = None):
        

        keys = jax.random.split(key, 12)

        print(f"keys shape: {keys.shape}")
        self.historical_steps = historical_steps
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self._center_embed = SingleInputEmbedding(node_dim, embed_dim, key=keys[0])
        self._nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], 
            out_channel=embed_dim,
            key=keys[1]
        )


        self.lin_q = eqx.nn.Linear(embed_dim, embed_dim, key=keys[2])
        self.lin_k = eqx.nn.Linear(embed_dim, embed_dim, key=keys[3])
        self.lin_v = eqx.nn.Linear(embed_dim, embed_dim, key=keys[4])
        self.lin_self = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])
        self.attn_dropout = eqx.nn.Dropout(dropout)

        self.lin_ih = eqx.nn.Linear(embed_dim, embed_dim, key=keys[6])
        self.lin_hh = eqx.nn.Linear(embed_dim, embed_dim, key=keys[7])
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[8])
        self.proj_drop = eqx.nn.Dropout(dropout)

        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[9]),
            jax.nn.relu,
            eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[10]),
            eqx.nn.Dropout(dropout)
        ])

        # Key differences from PyTorch:
        # Uses jnp.ndarray instead of torch.Tensor
        # Initialization is explicit using JAX's random number generator
        # No need for nn.Parameter as Equinox automatically treats array attributes as parameters
        # Initialize BOS token with random values
        bos_key = jax.random.fold_in(keys[11], 0)
        self.bos_token = jax.random.normal(
            bos_key,
            shape=(self.historical_steps, self.embed_dim)
        ) * 0.02  # Scale factor similar to PyTorch's default initialization

    def __call__(self, x, edge_index, edge_attr, bos_mask, t: Optional[int] = None, rotate_mat=None, size=None):
        
        # CHeange type signature to [List] node features, edge features, 
        # Break the call function into three steps
        # 1. Center Embedding
        # 2. Neighbor Embedding
        # 3. Multi-Head Attention propagate messages

        # Apply message function on all neighbors (just a vmap)

        # After when you do the aggregation, just a sum over all the messages where you take an inner product with the row and the vector messages
        # Altertively you can use where. If its 1 or 0 and when if it's 0 then you just 0 it. 
        # Keep it in form [node, node, channel]
        # After applyig sum it will collopse to [node, channel] # Einops sum
        
        
        # 4. Feedforward
        # 5. Residual Connection
        # 6. Layer Norm

        
        if rotate_mat is None:
            print("x.shape", x.shape)
            center_embed = self._center_embed(x)
        else:
            # TODO why do we have to expand dims to match rotation matrix
            # Look at what the dimensions represent
            center_embed = self._center_embed(jnp.squeeze(jnp.matmul(jnp.expand_dims(x, -2), rotate_mat), axis=-2)) # Do this in EINOPS
            # instead of matmul do @ 
            # Do it as two einops operations #TODO
        # Apply bos mask
        center_embed = jnp.where(
            jnp.expand_dims(bos_mask, -1),
            self.bos_token[t], 
            center_embed
        ) # Test if this has same behavior as PyTorch container
        # Do this in EINOPS
        # Debug prints
        print(f"JAX center_embed shape: {center_embed.shape}")
        print(f"JAX center_embed first few values: {center_embed[0, :5]}")
        print(f"JAX center_embed mean: {jnp.mean(center_embed)}")

        # I am jsut going to create a [src,target,channel] tensor
        # Then I can just do a vmap over the edge_index and then sum over the src axis
        # Then I can just take the inner product with the target axis and then add it to the center_embed
        # This is the message function

         # Apply MHA block with residual connection
        #center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, rotate_mat, size)
        # Now lets create the mha block but in line


        # Create a message funiton

        # Message passing / attention computation
        # jax.vmap(jax.vmap(self.create_message, in_axes=(0,0,0,None)), in_axes=(0,0))
        # Edge float[array]


        # Create a message funiton
        messages = self.create_message(x, edge_index, edge_attr, rotate_mat)
        # Broadcast 
        # Position and rotation matrix
        # Messgage and propagation
        # vmap(message)*edge, pos, rot_mat=rot_mat

        # def propagate
        # accunulate via sun
        # Before this run the where function
        # Boolean mask
        # return einops.reduce(msgs, "tgt embed_dim -> embed_dim", "sum")

        # Acculumation step
        # Compute messages using dense operations
        messages = jnp.einsum('ijh,jhd->ihd', alpha, value)  # [N, heads, head_dim]

        output = output.reshape(-1, self.embed_dim)
        # Inline update function
        gate = jax.nn.sigmoid(self.lin_ih(output) + self.lin_hh(center_embed))
        output = output + gate * (self.lin_self(center_embed) - output)

        # Apply feedforward block with residual connection
        #center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        # Project output and add residual connection
        output = self.out_proj(output)
        output = self.proj_drop(output)
        center_embed = center_embed + output

        # Apply feedforward block with residual connection
        center_embed = self.norm1(center_embed)
        ff_output = self.mlp(center_embed)
        center_embed = center_embed + ff_output
        center_embed = self.norm2(center_embed)


        return center_embed
    


    def create_message(self, x, relpos2neighbor, rotate_mat):

        # Rotation matrix is a [2,2] tensor
        # All 2,2 tensors 



        # Create a message funiton
        # First rotate the relative position by the rotation matrix

                # Everything goes into message function
        center_rotate_mat = rotate_mat[edge_index[1]]
        x_rotated = jnp.matmul(x[edge_index[1]][:, None, :], center_rotate_mat)[:, 0, :]
        edge_rotated = jnp.matmul(edge_attr[:, None, :], center_rotate_mat)[:, 0, :]
        
        # 


        nbr_embed = self._nbr_embed([
            jnp.squeeze(jnp.matmul(jnp.expand_dims(x[edge_index[1]], -2), center_rotate_mat), axis=-2),
            jnp.squeeze(jnp.matmul(jnp.expand_dims(edge_attr, -2), center_rotate_mat), axis=-2)
        ])


        # All 
        # Create dense attention matrix [num_nodes, num_nodes, embed_dim]
        N = len(x)
        attention_mask = jnp.zeros((N, N), dtype=bool)
        attention_mask = attention_mask.at[edge_index[0], edge_index[1]].set(True)

        # Compute Q from center nodes, K/V from neighbor embeddings
        query = self.lin_q(center_embed[edge_index[0]]).reshape(self.num_heads, self.embed_dim // self.num_heads)  # [E, heads, head_dim] # This should be an einops operation 
        key = self.lin_k(nbr_embed).reshape(self.num_heads, self.embed_dim // self.num_heads)  # [E, heads, head_dim] # This should be an einops operation
        value = self.lin_v(nbr_embed).reshape(self.num_heads, self.embed_dim // self.num_heads)  # [E, heads, head_dim] # This should be an einops operation    

        # Compute attention scores for all pairs
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = jnp.sum(query * key, axis=-1) / scale  # [E, num_heads]
        
        # Mask out non-connected pairs and apply softmax
        alpha = jnp.where(attention_mask[..., None], alpha, -1e9)
        alpha = jax.nn.softmax(alpha, axis=1)
        alpha = self.attn_dropout(alpha)
        