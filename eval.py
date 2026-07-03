"""
SOL-Nav Evaluation Script.

Evaluates a trained SOL-Nav model on test/val splits, computes metrics,
generates visualization plots, and saves sample prompts + model outputs.

Usage:
    # Evaluate on val_unseen
    python eval.py --config configs/default.yaml --checkpoint outputs/solnav_qwen3_lora_multires/final

    # Evaluate and save samples
    python eval.py --config configs/default.yaml --checkpoint outputs/solnav_qwen3_lora_multires/final --save_samples --num_samples 20
"""

import os
import sys
import json
import argparse
from collections import Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding
from peft import PeftModel
from torch.utils.data import DataLoader

from sol_nav.utils.config import load_config
from sol_nav.data.dataset import build_solnav_dataset, tokenize_function, ACTION_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="SOL-Nav Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--split", type=str, default="val_unseen",
                        choices=["val_seen", "val_unseen", "train"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_samples", action="store_true",
                        help="Save sample prompts and model outputs")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to save")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    return parser.parse_args()


def load_model(checkpoint_dir, device="cuda"):
    """Load trained SOL-Nav model from checkpoint."""
    from sol_nav.models.solnav_model import SOLNavMultiStepClassifier

    config_path = os.path.join(checkpoint_dir, "model_config.pt")
    if os.path.exists(config_path):
        model_config = torch.load(config_path, map_location="cpu", weights_only=False)
    else:
        model_config = {}

    # Load full model (LoRA + prediction heads)
    model = SOLNavMultiStepClassifier.from_checkpoint(
        checkpoint_dir,
        device=device,
    )
    model.eval()

    return model, model_config, model_config.get("class_weights", None)


def run_inference(model, tokenizer, dataset, batch_size=32, max_length=4096,
                  device="cuda", max_samples=None):
    """Run inference with full SOLNavMultiStepClassifier model."""
    # Tokenize
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["prompt"],
        num_proc=8,
    )
    tokenized = tokenized.rename_column("action_chunk", "labels")

    if max_samples:
        tokenized = tokenized.select(range(min(max_samples, len(tokenized))))

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
    )

    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Running inference on {len(tokenized)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()

            # Forward pass (inference mode - no labels needed for prediction)
            outputs = model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sequence_embedding = sum_embeddings / sum_mask

            logits_list = []
            for head in model.prediction_heads:
                step_logits = head(sequence_embedding)
                logits_list.append(step_logits)
            logits = torch.stack(logits_list, dim=1)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size}/{len(tokenized)}")

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return all_preds, all_labels, all_probs


def compute_evaluation_metrics(preds, labels, num_labels=4, num_steps=4):
    """Compute comprehensive evaluation metrics."""
    metrics = {}
    total_acc = 0.0

    for step in range(num_steps):
        step_labels = labels[:, step]
        step_preds = preds[:, step]

        valid_mask = step_labels != -100
        if not np.any(valid_mask):
            metrics[f"step{step}_acc"] = 0.0
            continue

        step_labels_v = step_labels[valid_mask]
        step_preds_v = step_preds[valid_mask]

        acc = accuracy_score(step_labels_v, step_preds_v)
        metrics[f"step{step}_acc"] = acc
        total_acc += acc

        # Per-class metrics
        report = classification_report(
            step_labels_v, step_preds_v,
            labels=list(range(num_labels)),
            output_dict=True,
            zero_division=0,
        )

        for cls_id in range(num_labels):
            cls_str = str(cls_id)
            cls_name = ACTION_NAMES.get(cls_id, f"class_{cls_id}")
            metrics[f"step{step}_precision_{cls_name}"] = report[cls_str]["precision"]
            metrics[f"step{step}_recall_{cls_name}"] = report[cls_str]["recall"]
            metrics[f"step{step}_f1_{cls_name}"] = report[cls_str]["f1-score"]

        # Overall per-step metrics
        metrics[f"step{step}_macro_precision"] = report["macro avg"]["precision"]
        metrics[f"step{step}_macro_recall"] = report["macro avg"]["recall"]
        metrics[f"step{step}_macro_f1"] = report["macro avg"]["f1-score"]

    metrics["mean_step_acc"] = total_acc / num_steps

    # First-step accuracy (primary metric)
    if "step0_acc" in metrics:
        metrics["first_step_acc"] = metrics["step0_acc"]

    return metrics


def plot_confusion_matrices(preds, labels, num_labels=4, num_steps=4,
                            output_dir=".", action_names=None):
    """Generate confusion matrix plots for each step."""
    if action_names is None:
        action_names = ACTION_NAMES

    fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 4))
    if num_steps == 1:
        axes = [axes]

    for step in range(num_steps):
        step_labels = labels[:, step]
        step_preds = preds[:, step]

        valid_mask = step_labels != -100
        step_labels_v = step_labels[valid_mask]
        step_preds_v = step_preds[valid_mask]

        cm = confusion_matrix(step_labels_v, step_preds_v,
                              labels=list(range(num_labels)))

        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[action_names.get(i, str(i)) for i in range(num_labels)],
            yticklabels=[action_names.get(i, str(i)) for i in range(num_labels)],
            ax=axes[step],
        )
        acc = accuracy_score(step_labels_v, step_preds_v)
        axes[step].set_title(f"Step {step} (Acc: {acc:.3f})")
        axes[step].set_ylabel("True")
        axes[step].set_xlabel("Predicted")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to {path}")


def plot_action_distribution(labels, preds, num_labels=4, num_steps=4,
                              output_dir=".", action_names=None):
    """Plot action distribution comparison between labels and predictions."""
    if action_names is None:
        action_names = ACTION_NAMES

    fig, axes = plt.subplots(1, num_steps, figsize=(5 * num_steps, 4))
    if num_steps == 1:
        axes = [axes]

    for step in range(num_steps):
        step_labels = labels[:, step]
        step_preds = preds[:, step]

        valid_mask = step_labels != -100
        step_labels_v = step_labels[valid_mask]
        step_preds_v = step_preds[valid_mask]

        label_counts = Counter(step_labels_v)
        pred_counts = Counter(step_preds_v)

        classes = list(range(num_labels))
        label_vals = [label_counts.get(c, 0) for c in classes]
        pred_vals = [pred_counts.get(c, 0) for c in classes]

        x = np.arange(num_labels)
        width = 0.35

        axes[step].bar(x - width/2, label_vals, width, label="Ground Truth", alpha=0.8)
        axes[step].bar(x + width/2, pred_vals, width, label="Predictions", alpha=0.8)
        axes[step].set_xticks(x)
        axes[step].set_xticklabels([action_names.get(c, str(c)) for c in classes], rotation=45)
        axes[step].set_title(f"Step {step}")
        axes[step].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "action_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved action distributions to {path}")


def plot_metrics_summary(metrics, output_dir="."):
    """Plot summary bar chart of key metrics."""
    key_metrics = {
        k: v for k, v in metrics.items()
        if k.startswith("step") and k.endswith("_acc")
    }
    key_metrics["mean_step_acc"] = metrics.get("mean_step_acc", 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(key_metrics.keys())
    values = list(key_metrics.values())

    colors = ["#2196F3"] * len(names)
    if "mean_step_acc" in names:
        colors[names.index("mean_step_acc")] = "#4CAF50"

    bars = ax.bar(names, values, color=colors, alpha=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Accuracy")
    ax.set_title("SOL-Nav Evaluation Metrics")
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "metrics_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics summary to {path}")


def save_samples(dataset, preds, labels, probs, num_samples, output_dir,
                 action_names=None):
    """Save sample prompts and model predictions."""
    if action_names is None:
        action_names = ACTION_NAMES

    samples = []
    n = min(num_samples, len(dataset), len(preds))

    # Select diverse samples: some correct, some incorrect
    correct_mask = np.all(preds[:len(dataset)] == labels[:len(dataset)], axis=1)
    correct_indices = np.where(correct_mask)[0][:n//2]
    incorrect_indices = np.where(~correct_mask)[0][:n - len(correct_indices)]
    sample_indices = np.concatenate([correct_indices, incorrect_indices])

    if len(sample_indices) < n:
        remaining = [i for i in range(len(dataset)) if i not in sample_indices]
        sample_indices = np.concatenate([sample_indices, remaining[:n - len(sample_indices)]])

    sample_indices = sample_indices[:n]

    for idx in sample_indices:
        sample = dataset[int(idx)]
        gt_actions = [action_names.get(a, str(a)) for a in labels[int(idx)]]
        pred_actions = [action_names.get(a, str(a)) for a in preds[int(idx)]]
        correct = bool(np.array_equal(preds[int(idx)], labels[int(idx)]))

        samples.append({
            "index": int(idx),
            "prompt": sample["prompt"],
            "ground_truth_actions": gt_actions,
            "predicted_actions": pred_actions,
            "ground_truth_labels": labels[int(idx)].tolist(),
            "predicted_labels": preds[int(idx)].tolist(),
            "prediction_probs": probs[int(idx)].tolist() if probs is not None else None,
            "correct": correct,
        })

    path = os.path.join(output_dir, "sample_predictions.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} sample predictions to {path}")

    # Also save a human-readable version
    txt_path = os.path.join(output_dir, "sample_predictions.txt")
    with open(txt_path, "w") as f:
        for s in samples:
            f.write(f"{'='*80}\n")
            f.write(f"Sample #{s['index']} | {'CORRECT' if s['correct'] else 'WRONG'}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Prompt:\n{s['prompt'][:2000]}\n\n")
            f.write(f"Ground Truth: {s['ground_truth_actions']}\n")
            f.write(f"Predicted:    {s['predicted_actions']}\n\n")
    print(f"Saved readable samples to {txt_path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_LABELS = model_cfg["num_labels"]
    NUM_CHUNK = model_cfg["num_steps"]
    MAX_LENGTH = model_cfg["max_length"]
    GRID_R = data_cfg["grid_resolutions"]["current"]
    HF_CACHE = model_cfg["hf_cache_dir"]

    # Output directory
    output_dir = args.output_dir or os.path.join(args.checkpoint, f"eval_{args.split}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"SOL-Nav Evaluation")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model...")
    model, model_config, class_weights = load_model(args.checkpoint, DEVICE)

    # Load dataset
    print("Loading dataset...")
    USE_MULTIRES = data_cfg.get("use_multires", True)

    if USE_MULTIRES:
        cache_tag = f"multires_L{data_cfg['history']['long_term_steps']}x{data_cfg['grid_resolutions']['long_term']}_S{data_cfg['history']['short_term_steps']}x{data_cfg['grid_resolutions']['short_term']}_C{data_cfg['grid_resolutions']['current']}_chunk{NUM_CHUNK}"
    else:
        num_his = data_cfg.get("num_his", 1)
        cache_tag = f"single_grid{GRID_R}_chunk{NUM_CHUNK}_his{num_his}"

    CACHE_BASE = data_cfg.get("cache_dir", "data/cache")
    if args.split == "train":
        data_dir = os.path.join(data_cfg["data_root"], "train", str(GRID_R))
    elif args.split == "val_seen":
        data_dir = os.path.join(data_cfg["data_root"], "val_seen", str(GRID_R))
    else:
        data_dir = os.path.join(data_cfg["data_root"], "val_unseen", str(GRID_R))

    cache_dir = os.path.join(CACHE_BASE, f"{args.split}_{cache_tag}")
    dataset = build_solnav_dataset(
        data_dir, cache_dir,
        num_grid_r=GRID_R, num_grid_c=GRID_R,
        num_chunk=NUM_CHUNK,
        long_term_steps=data_cfg.get("history", {}).get("long_term_steps", 16),
        short_term_steps=data_cfg.get("history", {}).get("short_term_steps", 2),
        current_steps=data_cfg.get("history", {}).get("current_steps", 1),
        long_term_res=data_cfg.get("grid_resolutions", {}).get("long_term", 2),
        short_term_res=data_cfg.get("grid_resolutions", {}).get("short_term", 4),
        current_res=data_cfg.get("grid_resolutions", {}).get("current", 6),
        num_his=data_cfg.get("num_his", 1),
        use_multires=USE_MULTIRES,
    )

    # Run inference
    print("Running inference...")
    preds, labels, probs = run_inference(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        batch_size=args.batch_size,
        max_length=MAX_LENGTH,
        device=DEVICE,
        max_samples=args.max_eval_samples,
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_evaluation_metrics(preds, labels, NUM_LABELS, NUM_CHUNK)

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({args.split})")
    print(f"{'='*60}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")

    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {os.path.join(output_dir, 'metrics.json')}")

    # Generate plots
    print("\nGenerating plots...")
    plot_confusion_matrices(preds, labels, NUM_LABELS, NUM_CHUNK, output_dir)
    plot_action_distribution(labels, preds, NUM_LABELS, NUM_CHUNK, output_dir)
    plot_metrics_summary(metrics, output_dir)

    # Save samples
    if args.save_samples:
        print("\nSaving sample predictions...")
        save_samples(dataset, preds, labels, probs, args.num_samples, output_dir)

    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
