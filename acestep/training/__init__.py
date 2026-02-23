"""
ACE-Step Training Module

This module provides LoRA training functionality for ACE-Step models,
including dataset building, audio labeling, and training utilities.
"""

from acestep.training.dataset_builder import DatasetBuilder, AudioSample
from acestep.training.configs import LoRAConfig, LoKRConfig, TrainingConfig
from acestep.training.configs import detect_best_precision, precision_to_dtype, precision_to_fabric_str
from acestep.training.lora_utils import (
    inject_lora_into_dit,
    save_lora_weights,
    load_lora_weights,
    merge_lora_weights,
    check_peft_available,
    check_model_quantized,
    upcast_model_for_training,
)
from acestep.training.data_module import (
    # Preprocessed (recommended)
    PreprocessedTensorDataset,
    PreprocessedDataModule,
    collate_preprocessed_batch,
    # Legacy (raw audio)
    AceStepTrainingDataset,
    AceStepDataModule,
    collate_training_batch,
    load_dataset_from_json,
)
from acestep.training.trainer import LoRATrainer, PreprocessedLoRAModule, LIGHTNING_AVAILABLE
from acestep.training.gpu_monitor import GPUMonitor, detect_gpu, get_available_vram_mb
from acestep.training.configs import list_presets, load_preset, apply_preset, auto_select_preset

# Optional: LoKr support via LyCORIS
try:
    from acestep.training.lokr_utils import (
        inject_lokr_into_dit,
        save_lokr_weights,
        load_lokr_weights,
        merge_lokr_weights,
        check_lycoris_available,
    )
except ImportError:
    pass  # LyCORIS not installed — LoKr features not available

def check_lightning_available():
    """Check if Lightning Fabric is available."""
    return LIGHTNING_AVAILABLE

__all__ = [
    # Dataset Builder
    "DatasetBuilder",
    "AudioSample",
    # Configs
    "LoRAConfig",
    "LoKRConfig",
    "TrainingConfig",
    # LoRA Utils
    "inject_lora_into_dit",
    "save_lora_weights",
    "load_lora_weights",
    "merge_lora_weights",
    "check_peft_available",
    "check_model_quantized",
    "upcast_model_for_training",
    # Precision Utils
    "detect_best_precision",
    "precision_to_dtype",
    "precision_to_fabric_str",
    # Data Module (Preprocessed - Recommended)
    "PreprocessedTensorDataset",
    "PreprocessedDataModule",
    "collate_preprocessed_batch",
    # Data Module (Legacy)
    "AceStepTrainingDataset",
    "AceStepDataModule",
    "collate_training_batch",
    "load_dataset_from_json",
    # Trainer
    "LoRATrainer",
    "PreprocessedLoRAModule",
    "check_lightning_available",
    "LIGHTNING_AVAILABLE",
    # GPU Monitor
    "GPUMonitor",
    "detect_gpu",
    "get_available_vram_mb",
    # Presets
    "list_presets",
    "load_preset",
    "apply_preset",
    "auto_select_preset",
]
