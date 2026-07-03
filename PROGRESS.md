# SOL-Nav Development Progress

## Completed Modules
- Project structure setup
- Text builder: multi-resolution prompt format matching supplement material
- Dataset: loading l2am_r2r_v3 format with multi-resolution grid downsampling
- Model: Qwen3-Embedding-0.6B + LoRA + multi-step classification heads
- Training script: full pipeline with wandb logging, multi-GPU support
- Evaluation script: inference, metrics, visualization, sample saving
- Configuration: optimized for 4x RTX 4090 (24GB VRAM)

## In Progress
- Training run on 4x RTX 4090 with torchrun

## To Do
- [ ] Verify training completes successfully
- [ ] Run evaluation on val_unseen split
- [ ] Generate visualization plots and sample predictions
- [ ] Final documentation update (README.md)
- [ ] Git commit with all changes

## Key Design Decisions
- Multi-resolution grid: 6x6 current, 4x4 short-term (2 frames), 2x2 long-term (16 frames)
- Prompt format matches supplement material exactly (including "### Structured Obervation:" typo)
- Mean pooling (not CLS) for robust sequence embedding
- bf16 precision training (RTX 4090 compatible)
- LoRA rank=16, alpha=32 (matching reference)
- batch_size=4 per GPU with gradient_accumulation=4 (effective batch=64 with 4 GPUs)

## Known Issues
- (none yet)
