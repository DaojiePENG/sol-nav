# SOL-Nav: Structured Observation Language for Efficient and Generalizable Vision-Language Navigation

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.27577-b31b1b)](https://arxiv.org/abs/2603.27577)
[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)](https://www.python.org/)

Official implementation of **SOL-Nav**, a Vision-Language Navigation (VLN) framework that converts egocentric RGB-D visual observations into compact structured language descriptions, enabling efficient and generalizable navigation via pure pre-trained language models (PLMs).

## Project Overview

SOL-Nav translates egocentric RGB-D observations into structured textual descriptions (semantic, color, depth information in N×N grids) and concatenates this with language instructions as pure language input to a PLM (Qwen3-Embedding-0.6B). This eliminates the need for visual encoders and leverages the full reasoning capabilities of pre-trained language models.

### Key Results (R2R-CE Val-Unseen)

| Metric | Value |
|--------|-------|
| First-step Accuracy | **73.15%** |
| Mean Step Accuracy | **66.31%** |
| Macro F1 (Step 0) | **0.4313** |
| Training Steps | 2,500 |
| Model Parameters | 600M (6.7M trainable via LoRA) |

## Quick Start

### Prerequisites

- Python 3.8+ (tested with 3.10)
- PyTorch 2.0+ with CUDA
- 4x NVIDIA RTX 4090 (24GB VRAM each) or equivalent
- Conda environment: `env_transformer_eval`

### Installation

```bash
# Clone the repository
git clone https://github.com/DaojiePENG/sol-nav.git
cd sol-nav

# Activate existing conda environment
conda activate env_transformer_eval

# Install dependencies (if not already installed)
pip install -r requirements.txt
# or
pip install torch transformers peft datasets accelerate scikit-learn \
    numpy pyyaml wandb tqdm matplotlib seaborn pandas
```

### Verify Installation

```bash
# Test model loading and forward pass
python -c "
import sys; sys.path.insert(0, '.')
from sol_nav.models.solnav_model import SOLNavMultiStepClassifier
import torch
model = SOLNavMultiStepClassifier(
    'Qwen/Qwen3-Embedding-0.6B', num_labels=4,
    class_weights=torch.ones(4), num_steps=4,
    cache_dir='data/hf_model_cache'
)
print(f'Model loaded: {model.embedding_dim}d, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')
"
```

## Dataset

### Format

The dataset is in `l2am_r2r_v3` format:
```
data/l2am_r2r_v3/
├── train/6/          # Training episodes (6x6 grid)
│   └── merged_part_*.json
├── val_seen/6/       # Validation seen episodes
│   └── merged_part_*.json
└── val_unseen/6/     # Validation unseen episodes
    └── merged_part_*.json
```

Each JSON file contains `{"episodes": [...]}` with episodes having:
- `episode_id`, `scene_id`, `instruction`
- `frames`: list of frame dicts with:
  - `time_step`, `semantic_patches`, `depth_patches`, `color_patches`
  - `agent_position`, `action` (0=stop, 1=turn_left, 2=turn_right, 3=move_forward)

### Multi-Resolution Prompt Format

SOL-Nav uses multi-resolution grids following the paper:
- **Long-term history**: 16 frames at 2×2 grid (oldest observations)
- **Short-term history**: 2 frames at 4×4 grid (recent observations)
- **Current observation**: 1 frame at 6×6 grid (full resolution)

Example prompt structure:
```
### System Description:
You are a robot that can turn left or right by a specific degree, move forward a certain distance, or stop...

### Structured Obervation:

[Time Step -18] Long Observation Grid:
[0,0]: depth=2.31, semantic=ceiling, color=light_gray; [0,1]: depth=2.31, semantic=ceiling, color=light_gray
[1,0]: depth=2.98, semantic=ceiling, color=gray; [1,1]: depth=2.98, semantic=ceiling, color=light_gray

[Time Step -17] Long Observation Grid:
...

[Time Step -2] Short Observation Grid:
[0,0]: depth=2.11, semantic=wall, color=yellow; [0,1]: ...
[1,0]: depth=2.45, semantic=wall, color=light_gray; [1,1]: ...
...

[Time Step 0] Current Observation Grid:
[0,0]: depth=2.08, semantic=window, color=gray; [0,1]: ...
[1,0]: depth=2.96, semantic=wall, color=gray; [1,1]: ...
...

### Task Instruction:
Go around the right side of the center unit and stop by the right side doorway.
```

See `samples/example_prompts.txt` for complete examples.

## Training

### Configuration

Training configuration is in `configs/default.yaml`:

```yaml
model:
  name: "Qwen/Qwen3-Embedding-0.6B"
  max_length: 2500
  num_labels: 4
  num_steps: 4

lora:
  rank: 16
  alpha: 32
  dropout: 0.05

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  num_epochs: 5
  bf16: true
  gradient_checkpointing: true
```

### Single GPU Training

```bash
python train.py --config configs/default.yaml
```

### Multi-GPU Training (4x RTX 4090)

```bash
# Set required environment variables for RTX 4090
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=sol-nav
export TOKENIZERS_PARALLELISM=false

# Launch with torchrun
torchrun --nproc_per_node=4 --master_port=29500 train.py \
    --config configs/default.yaml
```

### Resume Training from Checkpoint

```bash
torchrun --nproc_per_node=4 --master_port=29500 train.py \
    --config configs/default.yaml \
    --resume_from_checkpoint outputs/solnav_qwen3_lora_multires/checkpoint-2500
```

### Training Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to config YAML (default: configs/default.yaml) |
| `--data_root` | Override dataset root path |
| `--output_dir` | Override output directory |
| `--batch_size` | Override batch size per GPU |
| `--num_epochs` | Override number of epochs |
| `--force_rebuild` | Force rebuild cached datasets |
| `--resume_from_checkpoint` | Resume from checkpoint path |
| `--multires` / `--no_multires` | Toggle multi-resolution mode |

### Monitoring with WandB

Training metrics are logged to WandB:
- Loss curves
- Per-step accuracy (step 0-3)
- Per-class recall and F1 scores
- Learning rate schedule
- Gradient norms

Access at: `https://wandb.ai/<your-entity>/sol-nav`

## Evaluation

### Evaluate on Val-Unseen

```bash
python eval.py \
    --config configs/default.yaml \
    --checkpoint outputs/solnav_qwen3_lora_multires/final \
    --split val_unseen \
    --output_dir outputs/solnav_qwen3_lora_multires/eval_val_unseen_final \
    --batch_size 32 \
    --save_samples \
    --num_samples 30
```

### Evaluation Options

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to model checkpoint directory |
| `--split` | Dataset split: val_seen, val_unseen, train |
| `--output_dir` | Output directory for results |
| `--batch_size` | Inference batch size |
| `--save_samples` | Save sample predictions |
| `--num_samples` | Number of samples to save |
| `--max_eval_samples` | Limit evaluation samples (for debugging) |

### Evaluation Outputs

The evaluation generates:
- `metrics.json`: All evaluation metrics
- `confusion_matrices.png`: Per-step confusion matrices
- `action_distributions.png`: Ground truth vs predicted distributions
- `metrics_summary.png`: Summary bar chart
- `sample_predictions.json`: Detailed sample predictions
- `sample_predictions.txt`: Human-readable predictions

### Key Metrics

- **First-step Accuracy (step0_acc)**: Accuracy of the first action prediction
- **Mean Step Accuracy**: Average accuracy across all 4 action steps
- **Macro F1**: F1 score averaged across all action classes
- **Per-class Recall**: Recall for each action (stop, turn_left, turn_right, move_forward)

## Model Architecture

```
SOLNavMultiStepClassifier
├── base_model: Qwen3-Embedding-0.6B (with LoRA)
│   └── LoRA: rank=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj]
├── mean_pooling: Sequence → fixed-size embedding
└── prediction_heads: 4× (LayerNorm → Linear → GELU → Dropout → Linear)
    └── Each head: embedding_dim → embedding_dim//2 → num_labels
```

- **Backbone**: Qwen3-Embedding-0.6B (~600M params)
- **LoRA**: 4.59M trainable parameters (0.76% of total)
- **Pooling**: Mean pooling over sequence (more robust than CLS for long prompts)
- **Loss**: Weighted cross-entropy (balanced class weights)
- **Precision**: bf16 for efficient training on RTX 4090

## Project Structure

```
sol-nav/
├── configs/
│   └── default.yaml          # Training/evaluation configuration
├── data/
│   ├── cache/                # Cached datasets
│   └── hf_model_cache/       # HuggingFace model cache
├── models/                   # Saved model checkpoints
│   └── solnav_qwen3_lora_multires/
│       ├── final/            # Final model checkpoint
│       └── checkpoint-*/     # Training checkpoints
├── outputs/
│   └── solnav_qwen3_lora_multires/
│       ├── eval_val_unseen_final/  # Evaluation results
│       └── samples/          # Training sample prompts
├── samples/
│   ├── example_prompts.json  # Example prompts (JSON)
│   └── example_prompts.txt   # Example prompts (readable)
├── sol_nav/
│   ├── __init__.py
│   ├── data/
│   │   └── dataset.py        # Dataset loading and processing
│   ├── models/
│   │   └── solnav_model.py   # SOL-Nav model implementation
│   ├── navigation/
│   │   └── __init__.py       # Action space definition
│   └── utils/
│       ├── config.py         # Configuration loader
│       ├── logging.py        # Logging utilities
│       └── text_builder.py   # Multi-resolution prompt builder
├── train.py                  # Training script
├── eval.py                   # Evaluation script
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or max_length:
```bash
python train.py --config configs/default.yaml --batch_size 1
```

Or edit `configs/default.yaml`:
```yaml
model:
  max_length: 2048  # Reduce from 2500
training:
  per_device_train_batch_size: 1
```

### RTX 4090 Communication Errors

Set NCCL flags:
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Disk Space Issues

The dataset cache (~40GB for 631K samples) requires significant disk space. To reduce:
1. Remove old checkpoints: `rm -rf outputs/*/checkpoint-[0-9]*`
2. Skip tokenized cache (already handled in code)
3. Use `--force_rebuild` only when needed

### Slow Dataset Building

The first run builds the dataset cache (~10-15 minutes for 631K samples). Subsequent runs load from cache instantly.

## Citation

```bibtex
@article{peng2026structured,
  title={Structured Observation Language for Efficient and Generalizable Vision-Language Navigation},
  author={Peng, Daojie and Ma, Fulong and Ma, Jun},
  journal={arXiv preprint arXiv:2603.27577},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Contact

For questions: Daojie.PENG@qq.com
