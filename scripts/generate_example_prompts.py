"""
Generate example prompts for documentation / supplement material.

Reads the dataset, picks N random samples, and writes them in both
JSON and human-readable TXT format to samples/.

Usage:
    python scripts/generate_example_prompts.py                   # default: 5 samples from train split
    python scripts/generate_example_prompts.py --n 10            # 10 samples
    python scripts/generate_example_prompts.py --split val_unseen  # from val_unseen
    python scripts/generate_example_prompts.py --seed 123        # different random seed
"""

import os
import sys
import json
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sol_nav.data.dataset import build_solnav_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate example prompts from dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--n", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--split", default="train", choices=["train", "val_seen", "val_unseen"],
                        help="Dataset split to sample from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", default="samples", help="Output directory")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    data_root = data_cfg["data_root"]
    grid_cfg = data_cfg.get("grid_resolutions", {})
    hist_cfg = data_cfg.get("history", {})
    use_multires = data_cfg.get("use_multires", True)

    GRID_R = grid_cfg.get("current", 6)
    GRID_C = grid_cfg.get("current", 6)
    NUM_CHUNK = config["model"]["num_steps"]
    long_term_steps = hist_cfg.get("long_term_steps", 16)
    short_term_steps = hist_cfg.get("short_term_steps", 2)
    current_steps = hist_cfg.get("current_steps", 1)
    long_term_res = grid_cfg.get("long_term", 2)
    short_term_res = grid_cfg.get("short_term", 4)
    current_res = grid_cfg.get("current", 6)
    num_his = data_cfg.get("num_his", 1)
    cache_base = data_cfg.get("cache_dir", "data/cache")

    # 与 train.py 保持一致的路径构造
    data_dir = os.path.join(data_root, args.split, str(GRID_R))
    if use_multires:
        cache_tag = f"multires_L{long_term_steps}x{long_term_res}_S{short_term_steps}x{short_term_res}_C{current_res}_chunk{NUM_CHUNK}"
    else:
        cache_tag = f"single_grid{GRID_R}_chunk{NUM_CHUNK}_his{num_his}"
    cache_dir = os.path.join(cache_base, f"{args.split}_{cache_tag}")

    print(f"Building dataset from {data_dir} (split={args.split}, multires={use_multires})...")
    print(f"Cache: {cache_dir}")
    dataset = build_solnav_dataset(
        data_dir=data_dir,
        cache_dir=cache_dir,
        num_grid_r=GRID_R,
        num_grid_c=GRID_C,
        num_chunk=NUM_CHUNK,
        long_term_steps=long_term_steps,
        short_term_steps=short_term_steps,
        current_steps=current_steps,
        long_term_res=long_term_res,
        short_term_res=short_term_res,
        current_res=current_res,
        num_his=num_his,
        use_multires=use_multires,
    )

    print(f"Total samples: {len(dataset)}")

    # Pick N random samples
    n = min(args.n, len(dataset))
    indices = dataset.shuffle(seed=args.seed).select(range(n))

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # ── JSON output ──
    json_data = []
    for i, idx in enumerate(range(n)):
        sample = indices[i]
        json_data.append({
            "index": i,
            "prompt": sample["prompt"],
            "action": sample["action"],
            "action_chunk": sample["action_chunk"],
            "prompt_length_chars": len(sample["prompt"]),
            "prompt_length_lines": sample["prompt"].count("\n") + 1,
        })

    json_path = os.path.join(args.output_dir, "example_prompts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON: {json_path}")

    # ── TXT output ──
    txt_path = os.path.join(args.output_dir, "example_prompts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in json_data:
            n_chars = item["prompt_length_chars"]
            n_lines = item["prompt_length_lines"]
            action = item["action"]
            chunk = item["action_chunk"]
            idx = item["index"]

            f.write("=" * 80 + "\n")
            f.write(f"Sample #{idx} | Action: {action} | Chunk: {chunk}\n")
            f.write(f"Length: {n_chars} chars, {n_lines} lines\n")
            f.write("=" * 80 + "\n")
            f.write(item["prompt"])
            f.write("\n\n")
    print(f"Saved TXT:  {txt_path}")

    print(f"\nDone! Generated {n} example prompts from '{args.split}' split.")


if __name__ == "__main__":
    main()
