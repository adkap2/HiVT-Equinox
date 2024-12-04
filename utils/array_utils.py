from typing import Union, Any
import jax.numpy as jnp
import numpy as np
import torch

def print_array_type(arr: Union[jnp.ndarray, np.ndarray, list], 
                    var_name: str = "",
                    show_shape: bool = True,
                    show_type_annotation: bool = True) -> None:
    """
    Prints array information in a format useful for type annotations.
    
    Args:
        arr: The array to analyze
        var_name: Name of the variable (optional)
        show_shape: Whether to show the shape
        show_type_annotation: Whether to show the type annotation format
    """
    def get_dtype_str(dtype):
        # Map numpy/jax dtypes to type annotation strings
        dtype_map = {
            'float32': 'Float',
            'float64': 'Float',
            'int32': 'Int',
            'int64': 'Int',
            'bool': 'Bool',
            'complex64': 'Complex',
            'complex128': 'Complex',
            # PyTorch specific dtypes
            'torch.float32': 'Float',
            'torch.float64': 'Float',
            'torch.int32': 'Int',
            'torch.int64': 'Int',
            'torch.bool': 'Bool',
            'torch.complex64': 'Complex',
            'torch.complex128': 'Complex',
        }
        return dtype_map.get(str(dtype), 'Any')

    def get_type_str(shape):
        if not shape:
            return get_dtype_str(dtype)
        return f"Array[{get_type_str(shape[1:])}]" * shape[0]

    # Handle traced arrays
    if isinstance(arr, torch.Tensor):
        shape = tuple(arr.shape)
        dtype = arr.dtype
        val = arr
        batch_dim = None
    elif hasattr(arr, 'val'):  # For Jax traced arrays
        shape = arr.val.shape
        batch_dim = getattr(arr, 'batch_dim', None)
        dtype = arr.val.dtype
        val = arr.val
    else:  # For regular arrays
        shape = arr.shape if hasattr(arr, 'shape') else np.array(arr).shape
        dtype = arr.dtype if hasattr(arr, 'dtype') else np.array(arr).dtype
        batch_dim = None
        val = arr
    
    # Build the output string
    output = []
    if var_name:
        output.append(f"Variable: {var_name}")
    if show_shape:
        output.append(f"Shape: {shape}")
        output.append(f"dtype: {dtype}")
        if batch_dim is not None:
            output.append(f"Batch dimension: {batch_dim}")
        dims = [f"dim_{i}={s}" for i, s in enumerate(shape)]
        output.append(f"Dimensions: [{', '.join(dims)}]")
        output.append(f"Values:\n{val}")
    
    if show_type_annotation:
        base_type = get_dtype_str(dtype)
        if batch_dim is not None:
            # For traced arrays with batch dimension
            remaining_dims = [str(s) for i, s in enumerate(shape) if i != batch_dim]
            type_annotation = f"{base_type}[Array, \"batch {' '.join(remaining_dims)}\"]"
        else:
            # For regular arrays
            type_annotation = f"{base_type}[Array, \"{' '.join(str(s) for s in shape)}\"]"
        output.append(f"Type Annotation: {type_annotation}")
        
    # Print the results
    print("\n".join(output))
    print("-" * 50)