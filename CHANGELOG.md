# Changelog

## 2026-02-13 — Initial Release

First public release of the ACE-Step LoRA Trainer + Captioner as a standalone project, separated from the Pinokio launcher.

### Repository Setup
- **Initial commit** — Extracted core training modules (`acestep/training/`), model handler, LLM inference, captioner, and Gradio UIs from the Pinokio-based ACE-Step project into a self-contained standalone repository
- 20,000+ lines of code across 26 files
- Published to GitHub at `github.com/Estylon/ace-lora-trainer`

### Bug Fixes
- **Fixed ACESTEP_PATH pointing to non-existent directory** — Was looking for `ACE-Step-1.5/` which doesn't exist in standalone repo. Changed to repo root so checkpoint discovery works correctly
- **Fixed captioner/transcriber hardcoded Pinokio paths** — Changed `DEFAULT_MODEL_PATH` from `D:/pinokio/drive/.../acestep-captioner` to HuggingFace model ID `ACE-Step/acestep-captioner` so models auto-download on first use
- **Fixed captioner rejecting HuggingFace model IDs** — `os.path.exists()` was blocking valid HF model IDs like `ACE-Step/acestep-captioner`. Added HF ID detection (org/model format) to allow them through to `from_pretrained()`
- **Fixed torchao breaking Windows installs** — torchao has no Windows wheels and requires `torch.compile` + Triton (neither available on Windows). Made it Linux-only in `requirements.txt` since it's only needed for INT8/FP8 quantization, not LoRA training
- **Fixed RTX 3060/4060 GPU preset** — Changed optimizer from `adamw8bit` to `adafactor` and scheduler from `cosine` to `constant_with_warmup`. Adafactor uses near-zero optimizer state memory, making it viable for 8GB VRAM where AdamW 8-bit still OOMs

### Dependency Fixes
- Added missing `pydub` (audio splitting) and `diskcache` (caching) to requirements
- Removed unused `xxhash` dependency
- Added missing `acestep/genres_vocab.txt` (178K genre vocabulary file needed for caption generation)
- Updated `.gitignore` with `.cache/`, `lora_projects/`, `datasets/*.json`

### New Features

#### Install & Launch Scripts
- **`install.bat`** (Windows) and **`install.sh`** (Linux/Mac) — One-click installer that creates virtual environment, installs all dependencies via `uv pip` (with pip fallback), and verifies critical packages (PyTorch, PEFT, Lightning, Gradio, Prodigy, Transformers)
- **`start.bat`** (Windows) and **`start.sh`** (Linux/Mac) — One-click launcher that activates venv and runs the trainer
- **Environment checks in `launch.py`** — Detects if running without a virtual environment or with missing packages (especially PEFT) and warns before proceeding. This prevents the silent failure where training falls back to full fine-tuning without LoRA (~35x larger checkpoints, much slower)

#### VRAM Monitor
- **Real-time GPU status** in the Training tab showing GPU name, VRAM used/free/total with color-coded indicator (green/yellow/red)
- **Other process detection** — Uses `nvidia-smi` to list other processes occupying the GPU (e.g., ACE-Step API) with their VRAM usage
- **Pre-training VRAM check** — When clicking "Start Training", automatically checks for insufficient free VRAM and warns about other GPU processes
- **Refresh button** for manual VRAM status updates

### Documentation
- **`TRAINING_GUIDE.md`** — Comprehensive ~950-line training guide covering complete workflow (7 steps from captioning to merge), all parameters with defaults/ranges/explanations, all 4 GPU presets with full parameter tables, optimizer/scheduler comparison tables, disk space requirements, security audit section, and troubleshooting guide
- **`LICENSE`** — Apache License 2.0
- **`README.md`** — Updated with accurate GPU preset table (including scheduler column), auto-download model documentation, simplified install instructions using new scripts, and links to training guide

### Architecture Notes
The standalone trainer is **100% self-sufficient** — all model loading and inference happens in-process via PyTorch/HuggingFace Transformers. No external API server is required. The Gradio UI handles everything: model initialization, dataset building, auto-labeling, preprocessing, training, and LoRA merging.
