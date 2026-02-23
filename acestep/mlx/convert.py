"""
Weight Conversion: PyTorch -> MLX

Converts ACE-Step model weights from PyTorch format to MLX arrays.

Handles:
- Conv1d weight axis swaps (PyTorch: [out, in, k] -> MLX: [out, k, in])
- ConvTranspose1d axis swaps
- Sequential module index remapping
- Rotary embedding skipping (recomputed in MLX)
- weight_norm fusion (weight_g + weight_v -> weight)
"""

import os
from typing import Dict, Optional
from pathlib import Path
from loguru import logger


def is_rotary_key(key: str) -> bool:
    """Check if a parameter key belongs to rotary embeddings (skip these)."""
    rotary_keywords = ("rotary_emb", "inv_freq", "cos_cached", "sin_cached")
    return any(kw in key for kw in rotary_keywords)


def convert_conv1d_weight(weight, key: str = ""):
    """Convert Conv1d weight from PyTorch to MLX layout.

    PyTorch Conv1d: [out_channels, in_channels, kernel_size]
    MLX Conv1d:     [out_channels, kernel_size, in_channels]
    """
    import mlx.core as mx
    if len(weight.shape) == 3:
        return mx.array(weight.swapaxes(1, 2))
    return mx.array(weight)


def convert_conv_transpose1d_weight(weight, key: str = ""):
    """Convert ConvTranspose1d weight from PyTorch to MLX layout.

    PyTorch ConvTranspose1d: [in_channels, out_channels, kernel_size]
    MLX ConvTranspose1d:     [out_channels, kernel_size, in_channels]
    """
    import mlx.core as mx
    if len(weight.shape) == 3:
        # Transpose: [in, out, k] -> [out, k, in]
        return mx.array(weight.transpose(1, 2, 0))
    return mx.array(weight)


def fuse_weight_norm(state_dict: dict, prefix: str) -> dict:
    """Fuse weight_norm parameters (weight_g + weight_v -> weight).

    Some VAE layers use weight_norm which stores:
    - weight_g: magnitude scalar [out, 1, ...]
    - weight_v: direction tensor [out, in, ...]

    We fuse them into: weight = weight_g * (weight_v / ||weight_v||)

    Args:
        state_dict: Full state dict
        prefix: Key prefix to check

    Returns:
        Updated state dict with fused weights
    """
    import numpy as np

    g_key = f"{prefix}.weight_g"
    v_key = f"{prefix}.weight_v"

    if g_key in state_dict and v_key in state_dict:
        g = state_dict[g_key]
        v = state_dict[v_key]

        # Normalize v along non-output dimensions
        v_norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
        v_norm = v_norm.reshape(g.shape)

        weight = g * (v / (v_norm + 1e-12))

        # Replace with fused weight
        state_dict[f"{prefix}.weight"] = weight
        del state_dict[g_key]
        del state_dict[v_key]

    return state_dict


def convert_dit_weights(
    pytorch_state_dict: dict,
    verbose: bool = False,
) -> dict:
    """Convert DiT decoder weights from PyTorch to MLX format.

    Args:
        pytorch_state_dict: PyTorch state dict (numpy arrays or tensors)
        verbose: If True, log each converted key

    Returns:
        Dictionary of MLX arrays ready for loading into MLX DiT model.
    """
    import mlx.core as mx
    import numpy as np

    mlx_state = {}
    skipped = 0

    for key, value in pytorch_state_dict.items():
        # Skip rotary embeddings (recomputed in MLX)
        if is_rotary_key(key):
            skipped += 1
            continue

        # Convert tensor to numpy
        if hasattr(value, 'numpy'):
            value = value.cpu().numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array(value)

        # Handle Conv1d layers
        if ".conv." in key or key.endswith(".conv.weight"):
            if "weight" in key and len(value.shape) == 3:
                mlx_state[key] = convert_conv1d_weight(value, key)
                if verbose:
                    logger.debug(f"Conv1d: {key} {value.shape} -> {mlx_state[key].shape}")
                continue

        # Default: direct conversion
        mlx_state[key] = mx.array(value)
        if verbose:
            logger.debug(f"Direct: {key} {value.shape}")

    logger.info(f"Converted {len(mlx_state)} DiT parameters ({skipped} rotary keys skipped)")
    return mlx_state


def convert_vae_weights(
    pytorch_state_dict: dict,
    verbose: bool = False,
) -> dict:
    """Convert VAE (AutoencoderOobleck) weights from PyTorch to MLX format.

    Handles weight_norm fusion, Conv1d/ConvTranspose1d axis swaps,
    and Snake1d parameter reshaping.

    Args:
        pytorch_state_dict: PyTorch state dict
        verbose: If True, log each converted key

    Returns:
        Dictionary of MLX arrays.
    """
    import mlx.core as mx
    import numpy as np

    # First pass: fuse all weight_norm parameters
    np_dict = {}
    for key, value in pytorch_state_dict.items():
        if hasattr(value, 'numpy'):
            np_dict[key] = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            np_dict[key] = value
        else:
            np_dict[key] = np.array(value)

    # Find all weight_norm prefixes
    wn_prefixes = set()
    for key in list(np_dict.keys()):
        if key.endswith(".weight_g"):
            wn_prefixes.add(key[:-9])  # Remove .weight_g

    for prefix in wn_prefixes:
        np_dict = fuse_weight_norm(np_dict, prefix)
        if verbose:
            logger.debug(f"Fused weight_norm: {prefix}")

    # Second pass: convert to MLX with axis swaps
    mlx_state = {}
    for key, value in np_dict.items():
        # Snake1d alpha parameters: reshape from [1, C, 1] -> [C]
        if "alpha" in key and len(value.shape) == 3:
            mlx_state[key] = mx.array(value.squeeze())
            if verbose:
                logger.debug(f"Snake1d: {key} {value.shape} -> {mlx_state[key].shape}")
            continue

        # Detect Conv1d / ConvTranspose1d by key patterns
        is_conv = "conv" in key.lower() and "weight" in key and len(value.shape) == 3
        is_transpose = "upsample" in key.lower() or "decoder" in key.lower()

        if is_conv:
            if is_transpose and value.shape[0] > value.shape[1]:
                # Likely ConvTranspose1d: [in, out, k]
                mlx_state[key] = convert_conv_transpose1d_weight(value, key)
            else:
                # Standard Conv1d: [out, in, k]
                mlx_state[key] = convert_conv1d_weight(value, key)
            if verbose:
                logger.debug(f"Conv: {key} {value.shape} -> {mlx_state[key].shape}")
            continue

        mlx_state[key] = mx.array(value)

    logger.info(f"Converted {len(mlx_state)} VAE parameters")
    return mlx_state


def save_mlx_weights(mlx_state: dict, output_path: str):
    """Save MLX weights to a safetensors-compatible file.

    Args:
        mlx_state: Dictionary of MLX arrays
        output_path: Path to save (e.g., "model.safetensors")
    """
    import mlx.core as mx

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mx.savez(output_path, **mlx_state)
    logger.info(f"Saved MLX weights to {output_path}")


def load_mlx_weights(input_path: str) -> dict:
    """Load MLX weights from file.

    Args:
        input_path: Path to saved weights

    Returns:
        Dictionary of MLX arrays
    """
    import mlx.core as mx

    data = mx.load(input_path)
    logger.info(f"Loaded {len(data)} MLX parameters from {input_path}")
    return dict(data)
