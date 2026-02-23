"""
MLX Backend for Apple Silicon

Provides 2-3x faster inference on Apple Silicon (M1/M2/M3/M4) Macs
using the MLX framework for hardware-accelerated neural network execution.

This module implements:
- DiT (Diffusion Transformer) decoder in pure MLX
- VAE (AutoencoderOobleck) encoder/decoder in pure MLX
- Weight conversion utilities from PyTorch to MLX
- Full generation loop with KV caching and optional mx.compile

Requirements:
    pip install mlx mlx-nn

Usage:
    from acestep.mlx import is_mlx_available, MLXInferenceEngine

    if is_mlx_available():
        engine = MLXInferenceEngine(model_path="path/to/model")
        audio = engine.generate(prompt="upbeat jazz song", duration=30)
"""

import platform
import sys


def is_mlx_available() -> bool:
    """Check if MLX is available on this system.

    MLX requires:
    1. macOS operating system
    2. Apple Silicon (arm64) processor
    3. mlx Python package installed

    Returns:
        True if MLX can be used for inference.
    """
    if sys.platform != "darwin":
        return False

    if platform.machine() != "arm64":
        return False

    try:
        import mlx.core  # noqa: F401
        import mlx.nn  # noqa: F401
        return True
    except ImportError:
        return False


def mlx_available() -> bool:
    """Alias for is_mlx_available() for backward compatibility."""
    return is_mlx_available()


# Lazy imports to avoid loading MLX on non-Apple platforms
def get_mlx_engine():
    """Get the MLX inference engine class.

    Returns:
        MLXInferenceEngine class, or raises ImportError.
    """
    if not is_mlx_available():
        raise ImportError(
            "MLX is not available. MLX requires macOS on Apple Silicon. "
            "Install with: pip install mlx mlx-nn"
        )
    from acestep.mlx.engine import MLXInferenceEngine
    return MLXInferenceEngine
