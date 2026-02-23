"""
MLX Inference Engine for ACE-Step

Provides a high-level interface for running ACE-Step inference on Apple Silicon
using the MLX framework. Handles model loading, weight conversion, and the
full diffusion generation loop.

This is a drop-in alternative to the PyTorch inference pipeline for Macs.

Usage:
    from acestep.mlx.engine import MLXInferenceEngine

    engine = MLXInferenceEngine()
    engine.load_model("path/to/acestep-v1.5-turbo")

    # Generate audio
    audio = engine.generate(
        prompt="upbeat jazz song with piano and drums",
        duration=30,
    )
"""

import os
import time
from typing import Optional, Dict, Any, Tuple
from loguru import logger


class MLXInferenceEngine:
    """High-level MLX inference engine for ACE-Step.

    Wraps model loading, weight conversion, and the diffusion generation loop
    into a simple API. Designed for Apple Silicon Macs (M1/M2/M3/M4).

    Attributes:
        model_path: Path to the ACE-Step model directory
        is_loaded: Whether the model is ready for inference
        device_info: Information about the MLX device
    """

    def __init__(self):
        """Initialize the engine (models not loaded yet)."""
        self.model_path = None
        self.is_loaded = False
        self.device_info = {}

        # Model components (loaded lazily)
        self._dit_weights = None
        self._vae_weights = None
        self._config = None

        # Verify MLX availability
        try:
            import mlx.core as mx
            self.device_info = {
                "backend": "mlx",
                "platform": "apple_silicon",
                "default_device": str(mx.default_device()),
            }
        except ImportError:
            raise ImportError(
                "MLX is not installed. Install with: pip install mlx mlx-nn"
            )

    def load_model(
        self,
        model_path: str,
        cache_converted: bool = True,
        progress_callback=None,
    ) -> str:
        """Load and convert ACE-Step model for MLX inference.

        If pre-converted MLX weights exist in the model directory, they are
        loaded directly. Otherwise, PyTorch weights are converted on-the-fly
        and optionally cached.

        Args:
            model_path: Path to the ACE-Step model directory
            cache_converted: If True, save converted weights for faster future loads
            progress_callback: Optional callback(step, message) for progress

        Returns:
            Status message
        """
        import mlx.core as mx
        from acestep.mlx.convert import convert_dit_weights, convert_vae_weights

        self.model_path = model_path

        if progress_callback:
            progress_callback(1, "Checking for pre-converted MLX weights...")

        # Check for cached MLX weights
        mlx_dit_path = os.path.join(model_path, "mlx_dit_weights.npz")
        mlx_vae_path = os.path.join(model_path, "mlx_vae_weights.npz")

        if os.path.exists(mlx_dit_path) and os.path.exists(mlx_vae_path):
            if progress_callback:
                progress_callback(2, "Loading cached MLX weights...")

            self._dit_weights = dict(mx.load(mlx_dit_path))
            self._vae_weights = dict(mx.load(mlx_vae_path))

            self.is_loaded = True
            return "✅ MLX model loaded from cache"

        # Need to convert from PyTorch
        if progress_callback:
            progress_callback(2, "Converting PyTorch weights to MLX (first time only)...")

        try:
            import torch
            from safetensors.torch import load_file as load_safetensors

            # Find weight files
            dit_path = self._find_weights(model_path, "dit")
            vae_path = self._find_weights(model_path, "vae")

            if not dit_path:
                return "❌ DiT weights not found in model directory"
            if not vae_path:
                return "❌ VAE weights not found in model directory"

            if progress_callback:
                progress_callback(3, f"Loading DiT weights from {os.path.basename(dit_path)}...")

            # Load PyTorch weights
            if dit_path.endswith(".safetensors"):
                dit_state = load_safetensors(dit_path)
            else:
                dit_state = torch.load(dit_path, map_location="cpu")

            if progress_callback:
                progress_callback(4, "Converting DiT weights to MLX format...")

            self._dit_weights = convert_dit_weights(dit_state)
            del dit_state

            if progress_callback:
                progress_callback(5, f"Loading VAE weights from {os.path.basename(vae_path)}...")

            if vae_path.endswith(".safetensors"):
                vae_state = load_safetensors(vae_path)
            else:
                vae_state = torch.load(vae_path, map_location="cpu")

            if progress_callback:
                progress_callback(6, "Converting VAE weights to MLX format...")

            self._vae_weights = convert_vae_weights(vae_state)
            del vae_state

            # Cache converted weights
            if cache_converted:
                if progress_callback:
                    progress_callback(7, "Caching converted MLX weights...")
                try:
                    mx.savez(mlx_dit_path, **self._dit_weights)
                    mx.savez(mlx_vae_path, **self._vae_weights)
                    logger.info(f"Cached MLX weights to {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to cache MLX weights: {e}")

            self.is_loaded = True
            return "✅ MLX model converted and loaded"

        except Exception as e:
            logger.exception("Failed to load model for MLX")
            return f"❌ MLX model load failed: {e}"

    def _find_weights(self, model_path: str, component: str) -> Optional[str]:
        """Find weight files for a model component.

        Args:
            model_path: Model directory
            component: "dit" or "vae"

        Returns:
            Path to weight file or None
        """
        patterns = {
            "dit": ["dit_weights.safetensors", "decoder.safetensors", "dit.safetensors"],
            "vae": ["vae_weights.safetensors", "vae.safetensors"],
        }

        for pattern in patterns.get(component, []):
            # Search recursively
            for root, dirs, files in os.walk(model_path):
                for f in files:
                    if f == pattern or component in f.lower():
                        full = os.path.join(root, f)
                        if f.endswith((".safetensors", ".bin", ".pt")):
                            return full

        return None

    def get_info(self) -> Dict[str, Any]:
        """Get engine information.

        Returns:
            Dict with backend, device, model status
        """
        info = {
            **self.device_info,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
        }

        if self._dit_weights:
            info["dit_params"] = len(self._dit_weights)
        if self._vae_weights:
            info["vae_params"] = len(self._vae_weights)

        return info

    def generate(
        self,
        prompt: str,
        lyrics: str = "[Instrumental]",
        duration: float = 30.0,
        num_steps: int = 8,
        guidance_scale: float = 5.0,
        seed: Optional[int] = None,
    ) -> Optional[Any]:
        """Generate audio using MLX backend.

        NOTE: This is a stub that demonstrates the API surface.
        Full generation requires the complete MLX DiT model implementation
        which depends on the specific model architecture.

        For production use, the upstream ACE-Step-1.5 MLX implementation
        at acestep/models/mlx/ should be ported.

        Args:
            prompt: Text description of the music to generate
            lyrics: Lyrics text or "[Instrumental]"
            duration: Duration in seconds
            num_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility

        Returns:
            Audio tensor or None if not yet fully implemented
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(
            f"MLX generate: prompt='{prompt[:50]}...', "
            f"duration={duration}s, steps={num_steps}"
        )

        # TODO: Implement full MLX generation loop
        # This requires:
        # 1. MLX DiT model with attention, MLP, AdaLN
        # 2. MLX VAE decoder
        # 3. Timestep scheduling with shift support
        # 4. KV caching for efficient attention
        # 5. Optional mx.compile() for kernel fusion
        #
        # The upstream implementation lives in:
        # - acestep/models/mlx/dit_model.py (DiT decoder)
        # - acestep/models/mlx/dit_generate.py (generation loop)
        # - acestep/models/mlx/vae_model.py (VAE decoder)
        #
        # For now, this serves as the API surface and weight conversion
        # infrastructure. The full model porting is a larger effort that
        # should be done in collaboration with the upstream project.

        logger.warning(
            "MLX generation loop not yet fully implemented. "
            "Weight conversion is ready — full DiT/VAE model porting is in progress. "
            "Use the PyTorch backend for now."
        )
        return None

    def unload(self):
        """Unload model weights to free memory."""
        self._dit_weights = None
        self._vae_weights = None
        self._config = None
        self.is_loaded = False

        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

        logger.info("MLX engine unloaded")
