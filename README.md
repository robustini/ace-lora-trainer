# ACE-Step LoRA Trainer + Captioner

A standalone Gradio UI for training LoRA adapters on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) music generation models, with an integrated audio captioner.

## Features

### LoRA / LoKr Training
- **GPU Presets** — One-click optimal settings for RTX 4090, 3090, 3080, 3060 + external JSON presets
- **Multiple Optimizers** — AdamW, AdamW 8-bit, Adafactor, Prodigy (auto-LR)
- **Multiple Schedulers** — Cosine, Cosine Restarts, Linear, Constant, Constant+Warmup
- **Attention Targeting** — Train self-attention, cross-attention, or both
- **LoRA + LoKr** — PEFT LoRA (default) or LyCORIS LoKr (Kronecker factorization)
- **Gradient Checkpointing** — Save ~40-60% VRAM at ~30% speed cost
- **Encoder Offloading** — Move encoder to CPU during training, save ~2-4GB VRAM
- **Two-Pass Preprocessing** — Split preprocessing into two passes for 8-12GB GPUs
- **Gradient Sensitivity Estimation** — Identify which layers matter most for your dataset
- **GPU Monitoring** — Real-time VRAM tracking with alerts during training
- **Random Crop Augmentation** — Full songs randomly cropped to 60s each epoch
- **Early Stopping** — MA5 smoothed loss tracking with configurable patience
- **Auto-Save Best Model** — Automatically saves the checkpoint with lowest loss
- **Min-SNR Loss Weighting** — Reduces easy-timestep dominance (Hang et al. 2023)
- **Resume Training** — Continue from any saved checkpoint
- **LoRA Merge** — Merge LoRA/LoKr into base model as standalone safetensors
- **TensorBoard** — Live loss/LR curves via Lightning Fabric integration

### Audio Captioner
- **AI Captioning** — Automatic music style description using Qwen2.5 Omni (11B)
- **Lyrics Transcription** — Structured lyrics with [Verse], [Chorus] tags (ACE-Step Transcriber)
- **Metadata Extraction** — BPM, key, time signature, duration, genre
- **Batch Processing** — Process entire directories of audio files
- **Whisper API** — Cloud lyrics transcription via OpenAI (optional)
- **ElevenLabs Scribe** — Cloud lyrics transcription via ElevenLabs (optional)
- **Gemini API** — Cloud audio analysis: caption + lyrics + metadata in one call (optional)

### Platform Support
- **CUDA** — NVIDIA GPUs (primary, fully tested)
- **MPS** — Apple Silicon via Metal Performance Shaders
- **MLX** — Apple Silicon acceleration (weight conversion ready, generation in progress)
- **XPU** — Intel Arc GPUs
- **CPU** — Fallback for any platform

## Quick Start

### 1. Install

```bash
# Clone this repo
git clone https://github.com/Estylon/ace-lora-trainer.git
cd ace-lora-trainer
```

**Windows:**
```
install.bat
```

**Linux/Mac:**
```bash
chmod +x install.sh start.sh
./install.sh
```

<details>
<summary>Manual installation</summary>

```bash
# Create virtual environment (uv preferred, or python fallback)
uv venv env          # If uv is installed
# python -m venv env # Fallback if uv is not installed

# Activate — Windows:
env\Scripts\activate
# Activate — Linux/Mac:
source env/bin/activate

# Install dependencies
uv pip install -r requirements.txt
# Or: pip install -r requirements.txt
```
</details>

> **⚠️ Important:** Always use the virtual environment! Without it, critical packages like PEFT will be missing and training will silently fall back to full fine-tuning (much slower and ~35x larger checkpoints).

### 2. Models

**ACE-Step (generation model)** — Auto-downloaded on first launch to `./checkpoints/`.

**Captioner & Transcriber** — Auto-downloaded from HuggingFace on first use:
- Captioner: `ACE-Step/acestep-captioner` (~22GB)
- Transcriber: `ACE-Step/acestep-transcriber` (~22GB)

### 3. Launch

**Windows:**
```
start.bat
```

**Linux/Mac:**
```bash
./start.sh
```

<details>
<summary>Manual launch / options</summary>

```bash
# Activate venv first!
# Windows: env\Scripts\activate
# Linux/Mac: source env/bin/activate

# Training UI (default)
python launch.py

# Captioner UI
python launch.py --mode caption

# Both UIs simultaneously
python launch.py --mode both
```
</details>

## Recommended Settings by GPU

| GPU | VRAM | Batch | Rank | Optimizer | Scheduler | VRAM Features |
|-----|------|-------|------|-----------|-----------|---------------|
| RTX 4090 / 5090 | 24GB+ | 3 | 64 | Prodigy | Cosine | None needed |
| RTX 3090 / 4080 | 16-24GB | 2 | 64 | Prodigy | Cosine | None needed |
| RTX 3080 / 4070 | 10-12GB | 1 | 32 | AdamW 8bit | Cosine | Grad Ckpt + Offload |
| RTX 3060 / 4060 | 8GB | 1 | 16 | Adafactor | Constant+Warmup | Grad Ckpt + Offload + Two-Pass Preprocess |

> **Note (8GB):** Use Two-Pass Preprocessing (splits VAE+Encoder into separate passes) to avoid OOM during tensor creation. Adafactor stores nearly zero optimizer state, making it the safest choice for low-VRAM training.

External JSON presets are available in `acestep/training/presets/` — load with `auto_select_preset()` for automatic GPU-matched configuration.

## Training Workflow

1. **Load Model** — Select and initialize an ACE-Step checkpoint
2. **Build Dataset** — Scan audio files, label with AI captioner / Whisper / Gemini / manual
3. **Preprocess** — Convert audio + labels to training tensors (standard or two-pass for low VRAM)
4. **Train** — Select GPU preset or configure manually, start training with live monitoring
5. **Export** — Export LoRA/LoKr or merge into base model

## Epoch Recommendations

| Songs | Recommended Epochs | Save Every |
|-------|-------------------|------------|
| 1-3 | 1500 | 200 |
| 4-6 | 1000 | 200 |
| 7-10 | 700 | 100 |
| 11-20 | 500 | 100 |
| 21-50 | 300 | 50 |
| 50+ | 200 | 50 |

With Early Stop enabled, training will automatically stop when loss plateaus.

## Project Structure

```
ace-lora-trainer/
├── install.bat / install.sh  # One-click installer (creates venv + deps)
├── start.bat / start.sh      # One-click launcher (activates venv + runs)
├── launch.py                 # Main launcher (with environment checks)
├── lora_training_ui.py       # Training Gradio UI
├── captioner_standalone.py   # Captioner + Whisper/ElevenLabs/Gemini backends
├── requirements.txt          # Python dependencies
├── acestep/                  # Core ACE-Step modules
│   ├── training/             # Training pipeline
│   │   ├── trainer.py        # LoRA/LoKr trainer (Fabric + basic)
│   │   ├── configs.py        # Configuration + preset loader
│   │   ├── data_module.py    # Data loading (symlink-safe)
│   │   ├── dataset_builder.py# Dataset management + two-pass preprocessing
│   │   ├── lora_utils.py     # LoRA injection/save/load
│   │   ├── lokr_utils.py     # LoKr injection/save/load (LyCORIS)
│   │   ├── gpu_monitor.py    # Real-time VRAM monitoring
│   │   ├── estimator.py      # Gradient sensitivity estimation
│   │   └── presets/          # VRAM-aware JSON training presets
│   │       ├── recommended.json
│   │       ├── vram_8gb.json / vram_12gb.json / vram_16gb.json / vram_24gb_plus.json
│   │       ├── quick_test.json
│   │       └── high_quality.json
│   ├── mlx/                  # Apple Silicon MLX backend
│   │   ├── __init__.py       # MLX availability detection
│   │   ├── convert.py        # PyTorch → MLX weight conversion
│   │   └── engine.py         # MLX inference engine
│   ├── handler.py            # Model handler (LoRA + LoKr inference)
│   ├── gpu_config.py         # GPU tier detection
│   └── ...                   # Other ACE-Step modules
├── checkpoints/              # Model weights (auto-downloaded)
├── datasets/                 # Training datasets
└── lora_output/              # Training output
```

## Documentation

For a complete, in-depth guide covering every parameter, feature, and workflow:

**[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** — Full training guide with parameter explanations, GPU presets, disk space requirements, security notes, and troubleshooting.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) — Base music generation model
- [Side-Step](https://github.com/koda-dernet/Side-Step) — Inspiration for advanced training features
- Built with [Claude Code](https://claude.ai)
