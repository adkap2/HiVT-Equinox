import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray



class AAEncoder(eqx.Module):
    # Embeddings
    center_embed: eqx.nn.Sequential  # φcenter
    nbr_embed: eqx.nn.Sequential    # φnbr
    parallel: bool
    historical_steps: int
    
    #key: Optional[PRNGKeyArray]

    # Attention projections
    query_proj: eqx.nn.Linear  # WQspace
    key_proj: eqx.nn.Linear    # WKspace
    value_proj: eqx.nn.Linear  # WVspace
    
    # Gate and self attention
    gate_proj: eqx.nn.Linear   # Wgate
    self_proj: eqx.nn.Linear   # Wself
    
    # Output MLP and norms
    mlp: eqx.nn.Sequential
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    
    def __init__(self, historical_steps, node_dim, edge_dim, embed_dim, num_heads=8, parallel=True, *, key: Optional[PRNGKeyArray] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 11)   
        print(f"keys shape: {keys.shape}")
        self.parallel = parallel
        self.historical_steps = historical_steps

        # Embeddings (MLPs)
        """self.center_embed = eqx.nn.Sequential([
            eqx.nn.Linear(node_dim, embed_dim, key=keys[0]),
            jax.nn.relu,
            eqx.nn.Linear(embed_dim, embed_dim, key=keys[1])
        ])"""
        self.center_embed = SingleInputEmbedding(node_dim, embed_dim, key=keys[0])
        
        self.nbr_embed = MultipleInputEmbedding(
            in_channels=[node_dim, edge_dim], 
            out_channel=embed_dim,
            key=keys[1]
        )


        #print(f"keys[4] shape: {keys[4].shape}")
        
        # Attention projections
        head_dim = embed_dim // num_heads
        self.query_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[4])
        self.key_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[5])
        self.value_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[6])
        
        # Gate and self attention
        self.gate_proj = eqx.nn.Linear(embed_dim * 2, embed_dim, key=keys[7])
        self.self_proj = eqx.nn.Linear(embed_dim, embed_dim, key=keys[8])
        
        # Output MLP
        self.mlp = eqx.nn.Sequential([
            eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[9]),
            jax.nn.relu,
            eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[10])
        ])
        #print(f"keys[10] shape: {keys[10].shape}")
        # Layer norms
        self.norm1 = eqx.nn.LayerNorm(embed_dim)
        self.norm2 = eqx.nn.LayerNorm(embed_dim)

    def __call__(self, x, edge_index, edge_attr, rotate_mat=None):
        batch_size = x.shape[1]
        
        # For x shape (3, 4), we should NOT reshape since it's already in the correct format
        # Just use x directly for the center embedding
        if hasattr(self, 'parallel') and self.parallel:
            if rotate_mat is not None:
                # Add dimension for matrix multiplication
                print("rotate_mat", rotate_mat)
                x_expanded = jnp.expand_dims(x, -2)
                # Apply rotation
                x_rotated = jnp.matmul(x_expanded, rotate_mat)
                center_embed = self.center_embed(jnp.squeeze(x_rotated, -2))
            else:
                print("no rotate_mat")
                print(f"x shape before center_embed: {x.shape}")
                # Reshape x to (historical_steps, batch, features)
                x_processed = x.reshape(self.historical_steps, -1, x.shape[-1])
                center_embed = self.center_embed(x_processed)
        else:
            print("not parallel")
            center_embed = self.center_embed(x)
        print(f"edge_index shape: {edge_index.shape}")
        # 2. Compute center embeddings (Equation 1)
        print(f"x shape before reshape: {x.shape}")

        #reshaped_x = x.reshape(-1, x.shape[-1])
        #print(f"x shape after reshape: {reshaped_x.shape}")
        # center_embed = jax.vmap(self.center_embed)(x)
        #print(f"center_embed shape: {center_embed.shape}")
        # 1. Compute center embeddings (Equation 1)
        print("center_embed", self.center_embed)
        # Use lambda function with vmap
        #center_embed = jax.vmap(lambda y: self.center_embed(y))(x_reshaped)
        #center_embed = jax.vmap(self.center_embed)(x.reshape(-1, x.shape[-1]))
        #print(f"center_embed shape: {center_embed.shape}")
        
        
        # center_embed = center_embed.reshape(x.shape[0], batch_size, -1)
        # 3. Compute neighbor embeddings (Equation 2)
        src_idx, dst_idx = edge_index[0], edge_index[1]
        nbr_features = jnp.concatenate([
            x[:, src_idx],  # neighbor trajectories
            edge_attr[None, :, :].repeat(x.shape[0], axis=0)  # relative positions
        ], axis=-1)
        print(f"nbr_features shape: {nbr_features.shape}")
        nbr_embed = jax.vmap(self.nbr_embed)(nbr_features.reshape(-1, nbr_features.shape[-1]))
        nbr_embed = nbr_embed.reshape(x.shape[0], -1, nbr_embed.shape[-1])
        
        # 4. Compute attention (Equations 3-5)
        q = self.query_proj(center_embed)[:, dst_idx]
        k = self.key_proj(nbr_embed)
        v = self.value_proj(nbr_embed)
        
        # Scaled dot-product attention
        scale = (q.shape[-1]) ** -0.5
        attn = jnp.sum(q * k, axis=-1) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        
        # 5. Compute messages
        messages = v * attn[..., None]
        
        # 6. Apply gating (Equations 6-7)
        messages = jax.ops.segment_sum(messages, dst_idx, center_embed.shape[1])
        gate_input = jnp.concatenate([center_embed, messages], axis=-1)
        gate = jax.nn.sigmoid(self.gate_proj(gate_input))
        
        out = gate * self.self_proj(center_embed) + (1 - gate) * messages
        
        # 7. Apply residual connections and layer norms
        out = center_embed + self.norm1(out)
        out = out + self.norm2(self.mlp(out))
        
        return out

class SingleInputEmbedding(eqx.Module):
    embed: eqx.nn.Sequential

    def __init__(self, in_channel: int, out_channel: int, *, key: Optional[PRNGKeyArray] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        
        self.embed = eqx.nn.Sequential([
            eqx.nn.Linear(in_channel, out_channel, key=keys[0]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(out_channel, out_channel, key=keys[1]),
            eqx.nn.Lambda(jax.nn.relu),
            eqx.nn.Linear(out_channel, out_channel, key=keys[2])
        ])

    def __call__(self, x: Array) -> Array:
        print(f"x shape before embed: {x.shape}")
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        print("x_flat shape:", x_flat.shape)

        out = self.embed(x_flat)
        print("out shape:", out.shape)
        return out.reshape(*original_shape[:-1], -1)
        #return self.embed(x)

class MultipleInputEmbedding(eqx.Module):
    module_list: list
    aggr_embed: eqx.nn.Sequential

    def __init__(self, in_channels: List[int], out_channel: int, *, key: Optional[PRNGKeyArray] = None):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        # Need keys for each input module (4 layers each) plus aggr_embed (3 layers)
        num_keys_needed = len(in_channels) * 2 + 2  # 2 Linear layers per input + 2 for aggr_embed
        keys = jax.random.split(key, num_keys_needed)
        key_idx = 0
        
        # Create module list equivalent
        self.module_list = []
        for in_channel in in_channels:
            module = eqx.nn.Sequential([
                eqx.nn.Linear(in_channel, out_channel, key=keys[key_idx]),
                eqx.nn.LayerNorm(out_channel),
                jax.nn.relu,
                eqx.nn.Linear(out_channel, out_channel, key=keys[key_idx + 1])
            ])
            self.module_list.append(module)
            key_idx += 2
        
        # Create aggregation embedding
        self.aggr_embed = eqx.nn.Sequential([
            eqx.nn.LayerNorm(out_channel),
            jax.nn.relu,
            eqx.nn.Linear(out_channel, out_channel, key=keys[key_idx]),
            eqx.nn.LayerNorm(out_channel)
        ])

    def __call__(self, 
                 continuous_inputs: List[Array], 
                 categorical_inputs: Optional[List[Array]] = None) -> Array:
        # Process continuous inputs
        processed_inputs = []
        for i, module in enumerate(self.module_list):
            processed_inputs.append(module(continuous_inputs[i]))
        
        # Sum processed inputs
        output = jnp.stack(processed_inputs).sum(axis=0)
        
        # Add categorical inputs if provided
        if categorical_inputs is not None:
            output += jnp.stack(categorical_inputs).sum(axis=0)
        
        return self.aggr_embed(output)