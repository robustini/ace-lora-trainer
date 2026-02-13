"""
Gradient Sensitivity Estimation for ACE-Step LoRA Training

Estimates which attention modules contribute most to the loss for a given dataset.
This helps users select the most impactful layers for targeted LoRA training,
potentially improving quality while reducing trainable parameters.

Inspired by Side-Step's gradient estimation approach.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Generator
from collections import defaultdict
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext as _nullcontext

from acestep.training.configs import TrainingConfig, precision_to_dtype
from acestep.training.data_module import PreprocessedDataModule


def estimate_gradient_sensitivity(
    dit_handler,
    tensor_dir: str,
    training_config: TrainingConfig,
    max_batches: int = 10,
    granularity: str = "layer",
    top_k: int = 20,
) -> Generator[Tuple[int, str, Optional[List[Dict]]], None, None]:
    """Estimate gradient sensitivity of attention modules for a dataset.

    Runs forward/backward passes on a few batches and accumulates gradient
    L2 norms per attention module. Modules with higher gradient norms are
    more "sensitive" to the dataset and may benefit most from LoRA training.

    Args:
        dit_handler: Initialized DiT handler with loaded model
        tensor_dir: Path to preprocessed tensor directory
        training_config: Training configuration (for model_type, precision, etc.)
        max_batches: Maximum number of batches to evaluate (more = more accurate)
        granularity: "module" for individual projections (q/k/v/o_proj),
                     "layer" for attention blocks as a whole
        top_k: Number of top modules to return

    Yields:
        Tuples of (progress_pct, status_message, results_or_none)
        Final yield includes the ranked results list.
    """
    model = dit_handler.model
    device = dit_handler.device
    dtype = dit_handler.dtype

    yield 0, "🔍 Setting up gradient estimation...", None

    # Validate tensor directory
    if not os.path.exists(tensor_dir):
        yield 100, f"❌ Tensor directory not found: {tensor_dir}", None
        return

    # Create data module
    data_module = PreprocessedDataModule(
        tensor_dir=tensor_dir,
        batch_size=training_config.batch_size,
        num_workers=0,
        pin_memory=False,
        max_latent_length=training_config.max_latent_length,
    )
    data_module.setup('fit')

    if len(data_module.train_dataset) == 0:
        yield 100, "❌ No valid samples found", None
        return

    train_loader = data_module.train_dataloader()
    total_batches = min(max_batches, len(train_loader))

    yield 5, f"📂 Loaded {len(data_module.train_dataset)} samples, will evaluate {total_batches} batches", None

    # Resolve precision
    resolved_precision = training_config.resolve_precision()
    train_dtype = precision_to_dtype(resolved_precision)

    # Find all attention modules in the decoder
    decoder = model.decoder
    attention_modules = {}  # name -> module

    for name, module in decoder.named_modules():
        if isinstance(module, nn.Linear):
            short_name = name.split(".")[-1]
            if short_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                attention_modules[name] = module

    if not attention_modules:
        yield 100, "❌ No attention modules found in decoder", None
        return

    yield 10, f"🎯 Found {len(attention_modules)} attention projection modules", None

    # Temporarily enable gradients on all attention modules
    original_requires_grad = {}
    for name, module in attention_modules.items():
        for pname, param in module.named_parameters():
            full_name = f"{name}.{pname}"
            original_requires_grad[full_name] = param.requires_grad
            param.requires_grad = True

    # Accumulate gradient norms
    grad_norms = defaultdict(float)
    batch_count = 0

    # Import timestep sampling functions
    from acestep.training.trainer import sample_discrete_timestep, sample_continuous_timestep

    # Determine autocast context
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    use_autocast = device_type in ("cuda", "xpu") and train_dtype != torch.float32

    try:
        decoder.train()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= total_batches:
                break

            progress_pct = 10 + int(80 * (batch_idx + 1) / total_batches)
            yield progress_pct, f"⏳ Processing batch {batch_idx + 1}/{total_batches}...", None

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            autocast_ctx = torch.autocast(device_type=device_type, dtype=train_dtype) if use_autocast else _nullcontext()

            with autocast_ctx:
                target_latents = batch["target_latents"]
                attention_mask = batch["attention_mask"]
                encoder_hidden_states = batch["encoder_hidden_states"]
                encoder_attention_mask = batch["encoder_attention_mask"]
                context_latents = batch["context_latents"]

                bsz = target_latents.shape[0]

                # Flow matching setup
                x1 = torch.randn_like(target_latents)
                x0 = target_latents

                # Sample timesteps
                if training_config.is_turbo:
                    t, r = sample_discrete_timestep(bsz, device, train_dtype)
                else:
                    t, r = sample_continuous_timestep(
                        bsz, device, train_dtype,
                        shift=training_config.shift,
                        num_steps=training_config.num_inference_steps,
                    )

                t_ = t.unsqueeze(-1).unsqueeze(-1)
                xt = t_ * x1 + (1.0 - t_) * x0

                # Forward pass
                decoder_outputs = decoder(
                    hidden_states=xt,
                    timestep=t,
                    timestep_r=r,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    context_latents=context_latents,
                )

                # Loss
                flow = x1 - x0
                loss = F.mse_loss(decoder_outputs[0], flow)

            # Backward
            loss = loss.float()
            loss.backward()

            # Accumulate gradient norms per module
            for name, module in attention_modules.items():
                for pname, param in module.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.data.float().norm(2).item()
                        grad_norms[name] += norm

            # Zero gradients
            decoder.zero_grad()
            batch_count += 1

    except Exception as e:
        logger.exception("Gradient estimation failed")
        yield 100, f"❌ Estimation failed: {str(e)}", None
        return
    finally:
        # Restore original requires_grad state
        for name, module in attention_modules.items():
            for pname, param in module.named_parameters():
                full_name = f"{name}.{pname}"
                if full_name in original_requires_grad:
                    param.requires_grad = original_requires_grad[full_name]
        decoder.zero_grad()
        decoder.eval()

    if batch_count == 0:
        yield 100, "❌ No batches processed", None
        return

    yield 92, "📊 Computing rankings...", None

    # Normalize by batch count
    for name in grad_norms:
        grad_norms[name] /= batch_count

    # Group by granularity
    if granularity == "layer":
        # Group by attention block (e.g., "layers.0.self_attn" from "layers.0.self_attn.q_proj")
        layer_norms = defaultdict(float)
        layer_counts = defaultdict(int)

        for name, norm in grad_norms.items():
            # Remove the projection suffix to get the attention block name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                layer_name = parts[0]
            else:
                layer_name = name
            layer_norms[layer_name] += norm
            layer_counts[layer_name] += 1

        # Average within each layer
        for layer_name in layer_norms:
            if layer_counts[layer_name] > 0:
                layer_norms[layer_name] /= layer_counts[layer_name]

        ranked = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
    else:
        # Module-level granularity
        ranked = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)

    # Normalize scores to [0, 1]
    max_score = ranked[0][1] if ranked else 1.0
    results = []
    for name, score in ranked[:top_k]:
        results.append({
            "module": name,
            "score": round(score / max_score, 4) if max_score > 0 else 0.0,
            "raw_norm": round(score, 6),
        })

    yield 100, f"✅ Estimation complete! Top {len(results)} modules ranked by gradient sensitivity.", results


def save_estimation_results(results: List[Dict], output_path: str) -> str:
    """Save estimation results to a JSON file.

    Args:
        results: List of module ranking dicts from estimate_gradient_sensitivity
        output_path: Path to save JSON file

    Returns:
        Path to saved file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Estimation results saved to {output_path}")
    return output_path
