"""
SOL-Nav Training Script.

Fine-tunes Qwen3-Embedding-0.6B with LoRA for multi-step action prediction
using structured observation prompts with multi-resolution grids.

Usage:
    # Single GPU
    python train.py --config configs/default.yaml

    # Multi-GPU (4x RTX 4090)
    torchrun --nproc_per_node=4 train.py --config configs/default.yaml
"""

import os
import sys
import json
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import concatenate_datasets, load_from_disk
from sklearn.metrics import classification_report, accuracy_score

from sol_nav.utils.config import load_config
from sol_nav.data.dataset import build_solnav_dataset, tokenize_function
from sol_nav.models.solnav_model import SOLNavMultiStepClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="SOL-Nav Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_his", type=int, default=None,
                        help="Number of history frames (single-res mode)")
    parser.add_argument("--grid_r", type=int, default=None,
                        help="Grid rows (current resolution)")
    parser.add_argument("--grid_c", type=int, default=None,
                        help="Grid columns (current resolution)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--force_rebuild", action="store_true",
                        help="Force rebuild cached datasets")
    parser.add_argument("--multires", action="store_true", default=None,
                        help="Use multi-resolution mode")
    parser.add_argument("--no_multires", action="store_true",
                        help="Disable multi-resolution mode")
    return parser.parse_args()


def compute_metrics_factory(num_chunk, num_labels):
    """Create compute_metrics function for Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        metrics = {}
        total_acc = 0.0

        for step in range(num_chunk):
            step_labels = labels[:, step]
            step_preds = preds[:, step]

            valid_mask = step_labels != -100
            if not np.any(valid_mask):
                for cls_id in range(num_labels):
                    metrics[f"step{step}_recall_class_{cls_id}"] = 0.0
                    metrics[f"step{step}_f1_class_{cls_id}"] = 0.0
                metrics[f"step{step}_acc"] = 0.0
                continue

            step_labels_v = step_labels[valid_mask]
            step_preds_v = step_preds[valid_mask]

            acc = accuracy_score(step_labels_v, step_preds_v)
            metrics[f"step{step}_acc"] = acc
            total_acc += acc

            report = classification_report(
                step_labels_v, step_preds_v,
                labels=list(range(num_labels)),
                output_dict=True,
                zero_division=0,
            )
            for cls_id in range(num_labels):
                cls_str = str(cls_id)
                metrics[f"step{step}_recall_class_{cls_id}"] = report[cls_str]["recall"]
                metrics[f"step{step}_f1_class_{cls_id}"] = report[cls_str]["f1-score"]

        metrics["mean_step_acc"] = total_acc / num_chunk

        if "step0_acc" in metrics:
            metrics["first_step_acc"] = metrics["step0_acc"]
            for cls_id in range(num_labels):
                metrics[f"first_step_recall_class_{cls_id}"] = metrics.get(f"step0_recall_class_{cls_id}", 0.0)
                metrics[f"first_step_f1_class_{cls_id}"] = metrics.get(f"step0_f1_class_{cls_id}", 0.0)

        return metrics

    return compute_metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir
    if args.model_name:
        cfg["model"]["name"] = args.model_name
    if args.num_epochs:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.learning_rate:
        cfg["training"]["learning_rate"] = args.learning_rate
    if args.batch_size:
        cfg["training"]["per_device_train_batch_size"] = args.batch_size
    if args.no_multires:
        cfg["data"]["use_multires"] = False
    elif args.multires:
        cfg["data"]["use_multires"] = True

    # Extract configs
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    wandb_cfg = cfg["wandb"]

    DATA_ROOT = data_cfg["data_root"]
    MODEL_NAME = model_cfg["name"]
    HF_CACHE = model_cfg["hf_cache_dir"]
    MAX_LENGTH = model_cfg["max_length"]
    NUM_LABELS = model_cfg["num_labels"]
    NUM_CHUNK = model_cfg["num_steps"]
    OUTPUT_DIR = train_cfg["output_dir"]
    GRID_R = args.grid_r or data_cfg["grid_resolutions"]["current"]
    GRID_C = args.grid_c or data_cfg["grid_resolutions"]["current"]
    USE_MULTIRES = data_cfg.get("use_multires", True)

    # Multi-resolution config
    long_term_steps = data_cfg.get("history", {}).get("long_term_steps", 16)
    short_term_steps = data_cfg.get("history", {}).get("short_term_steps", 2)
    current_steps = data_cfg.get("history", {}).get("current_steps", 1)
    long_term_res = data_cfg.get("grid_resolutions", {}).get("long_term", 2)
    short_term_res = data_cfg.get("grid_resolutions", {}).get("short_term", 4)
    current_res = data_cfg.get("grid_resolutions", {}).get("current", 6)

    # Single-res fallback
    num_his = args.num_his or data_cfg.get("num_his", 1)

    # Build cache tag
    if USE_MULTIRES:
        cache_tag = f"multires_L{long_term_steps}x{long_term_res}_S{short_term_steps}x{short_term_res}_C{current_res}_chunk{NUM_CHUNK}"
    else:
        cache_tag = f"single_grid{GRID_R}_chunk{NUM_CHUNK}_his{num_his}"

    # Data paths
    TRAIN_DIR = os.path.join(DATA_ROOT, "train", str(GRID_R))
    VAL_SEEN_DIR = os.path.join(DATA_ROOT, "val_seen", str(GRID_R))
    VAL_UNSEEN_DIR = os.path.join(DATA_ROOT, "val_unseen", str(GRID_R))

    CACHE_BASE = data_cfg.get("cache_dir", "data/cache")
    CACHE_DIR = os.path.join(CACHE_BASE, f"train_{cache_tag}")
    VAL_CACHE_DIR = os.path.join(CACHE_BASE, f"val_seen_{cache_tag}")
    VAL_U_CACHE_DIR = os.path.join(CACHE_BASE, f"val_unseen_{cache_tag}")

    # Setup wandb
    if wandb_cfg.get("enabled", True):
        os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "sol-nav")
        wandb_run_name = wandb_cfg.get("run_name", f"solnav-{cache_tag}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        wandb_run_name = "solnav-offline"

    print(f"{'='*60}")
    print(f"SOL-Nav Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Multi-resolution: {USE_MULTIRES}")
    if USE_MULTIRES:
        print(f"  Long-term: {long_term_steps} frames @ {long_term_res}x{long_term_res}")
        print(f"  Short-term: {short_term_steps} frames @ {short_term_res}x{short_term_res}")
        print(f"  Current: {current_steps} frame(s) @ {current_res}x{current_res}")
    else:
        print(f"  Grid: {GRID_R}x{GRID_C}, History: {num_his}")
    print(f"Action chunk: {NUM_CHUNK}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
        padding_side="right",
        clean_up_tokenization_spaces=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build datasets
    print("Building datasets...")
    train_ds = build_solnav_dataset(
        TRAIN_DIR, CACHE_DIR,
        num_grid_r=GRID_R, num_grid_c=GRID_C,
        num_chunk=NUM_CHUNK,
        long_term_steps=long_term_steps,
        short_term_steps=short_term_steps,
        current_steps=current_steps,
        long_term_res=long_term_res,
        short_term_res=short_term_res,
        current_res=current_res,
        num_his=num_his,
        use_multires=USE_MULTIRES,
        force_rebuild=args.force_rebuild,
    )
    val_seen_ds = build_solnav_dataset(
        VAL_SEEN_DIR, VAL_CACHE_DIR,
        num_grid_r=GRID_R, num_grid_c=GRID_C,
        num_chunk=NUM_CHUNK,
        long_term_steps=long_term_steps,
        short_term_steps=short_term_steps,
        current_steps=current_steps,
        long_term_res=long_term_res,
        short_term_res=short_term_res,
        current_res=current_res,
        num_his=num_his,
        use_multires=USE_MULTIRES,
        force_rebuild=args.force_rebuild,
    )
    val_unseen_ds = build_solnav_dataset(
        VAL_UNSEEN_DIR, VAL_U_CACHE_DIR,
        num_grid_r=GRID_R, num_grid_c=GRID_C,
        num_chunk=NUM_CHUNK,
        long_term_steps=long_term_steps,
        short_term_steps=short_term_steps,
        current_steps=current_steps,
        long_term_res=long_term_res,
        short_term_res=short_term_res,
        current_res=current_res,
        num_his=num_his,
        use_multires=USE_MULTIRES,
        force_rebuild=args.force_rebuild,
    )

    # Data augmentation: merge val splits into training
    aug_ratio = data_cfg.get("augment_ratio", 0.8)
    aug_ratio_u = data_cfg.get("augment_ratio_u", 0.6)

    n_vs = int(len(val_seen_ds) * aug_ratio)
    n_vu = int(len(val_unseen_ds) * aug_ratio_u)
    val_seen_sampled = val_seen_ds.shuffle(seed=42).select(range(n_vs))
    val_unseen_sampled = val_unseen_ds.shuffle(seed=42).select(range(n_vu))

    train_ds = concatenate_datasets([train_ds, val_seen_sampled, val_unseen_sampled])
    train_ds = train_ds.shuffle(seed=42)

    # Evaluation set (20% of val_seen)
    eval_ds = val_seen_ds.shuffle(seed=42).select(range(int(len(val_seen_ds) * 0.2)))

    print(f"Training samples: {len(train_ds)}")
    print(f"Evaluation samples: {len(eval_ds)}")
    print(f"Val unseen samples: {len(val_unseen_ds)}")

    # Save sample prompts for analysis
    samples_dir = os.path.join(OUTPUT_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_data = []
    for i in range(min(10, len(train_ds))):
        sample_data.append({
            "prompt": train_ds[i]["prompt"],
            "action": train_ds[i]["action"],
            "action_chunk": train_ds[i]["action_chunk"],
        })
    with open(os.path.join(samples_dir, "train_samples.json"), "w") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    # Tokenize
    tokenized_dir = os.path.join(OUTPUT_DIR, "tokenized")
    tok_train_path = os.path.join(tokenized_dir, "train")
    tok_eval_path = os.path.join(tokenized_dir, "eval")

    if os.path.exists(tok_train_path) and os.path.exists(tok_eval_path) and not args.force_rebuild:
        print("Loading tokenized datasets from cache...")
        tokenized_train = load_from_disk(tok_train_path)
        tokenized_eval = load_from_disk(tok_eval_path)
    else:
        print("Tokenizing datasets...")
        tokenized_train = train_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["prompt"],
            num_proc=16,
        )
        tokenized_eval = eval_ds.map(
            lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
            batched=True,
            remove_columns=["prompt"],
            num_proc=16,
        )

        tokenized_train = tokenized_train.rename_column("action_chunk", "labels")
        tokenized_eval = tokenized_eval.rename_column("action_chunk", "labels")

        os.makedirs(tokenized_dir, exist_ok=True)
        tokenized_train.save_to_disk(tok_train_path)
        tokenized_eval.save_to_disk(tok_eval_path)

    # Compute class weights from original (pre-tokenized) train_ds
    all_actions = train_ds["action"]
    unique_labels = np.unique(all_actions)
    num_labels = len(unique_labels)
    print(f"Action classes: {num_labels}, Distribution: {dict(zip(*np.unique(all_actions, return_counts=True)))}")

    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=unique_labels,
        y=all_actions,
    )
    class_weights = torch.tensor(class_weights_arr, dtype=torch.float32)
    print(f"Class weights: {class_weights}")

    # Print a sample prompt
    print(f"\n{'='*60}")
    print("Sample prompt (first 1000 chars):")
    print(train_ds[0]["prompt"][:1000])
    print(f"Action chunk: {train_ds[0]['action_chunk']}")
    print(f"{'='*60}\n")

    # Initialize model
    print(f"Loading model: {MODEL_NAME} with LoRA")
    model = SOLNavMultiStepClassifier(
        MODEL_NAME,
        num_labels=num_labels,
        class_weights=class_weights,
        num_steps=NUM_CHUNK,
        cache_dir=HF_CACHE,
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        lora_target_modules=lora_cfg["target_modules"],
    )

    # Parameter statistics
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M ({100*trainable/total:.4f}%)")

    # Training arguments
    # RTX 4090 supports bf16; batch_size=4 per GPU fits in 24GB with mean-pooled embeddings
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to="wandb" if wandb_cfg.get("enabled", True) else "none",
        run_name=wandb_run_name,
        seed=train_cfg["seed"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", True),
        remove_unused_columns=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
    )

    # Create Trainer
    compute_metrics = compute_metrics_factory(NUM_CHUNK, num_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    resume = args.resume_from_checkpoint or train_cfg.get("resume_from_checkpoint")
    if resume:
        print(f"Resuming from: {resume}")
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()

    # Save final model (LoRA weights + prediction heads + config)
    final_dir = os.path.join(OUTPUT_DIR, "final")
    model.save_checkpoint(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save({
        "class_weights": class_weights,
        "num_labels": num_labels,
        "num_steps": NUM_CHUNK,
        "grid_r": GRID_R,
        "grid_c": GRID_C,
        "use_multires": USE_MULTIRES,
        "long_term_steps": long_term_steps,
        "short_term_steps": short_term_steps,
        "current_steps": current_steps,
        "long_term_res": long_term_res,
        "short_term_res": short_term_res,
        "current_res": current_res,
    }, os.path.join(final_dir, "model_config.pt"))

    print(f"Training complete! Model saved to {final_dir}")

    # Run final evaluation on val_unseen
    print("\nRunning final evaluation on val_unseen split...")
    tokenized_val_unseen = val_unseen_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=["prompt"],
        num_proc=16,
    )
    tokenized_val_unseen = tokenized_val_unseen.rename_column("action_chunk", "labels")

    eval_results = trainer.evaluate(tokenized_val_unseen)
    print(f"\nVal-Unseen Results:")
    for k, v in sorted(eval_results.items()):
        print(f"  {k}: {v}")

    # Save eval results
    with open(os.path.join(OUTPUT_DIR, "eval_results_val_unseen.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
