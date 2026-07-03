"""
SOL-Nav Dataset Utilities.

Loads preprocessed R2R/RxR data (l2am_r2r_v3 format), builds structured
observation prompts with multi-resolution grids, and prepares action chunk
labels for training.

Data format:
  Each JSON file contains {"episodes": [...]}. Each episode has:
    - episode_id, scene_id, instruction, start_position, start_rotation
    - frames: list of frame dicts, each with:
        - time_step, semantic_patches, depth_patches, color_patches
        - agent_position, action

  Patches are stored as dict mapping "(i,j)" -> value.
  Grid resolution is 6x6 (36 patches per frame).

Multi-resolution history:
  - Long-term history: 16 frames at 2x2 grid (downsampled from 6x6)
  - Short-term history: 2 frames at 4x4 grid (downsampled from 6x6)
  - Current observation: 1 frame at 6x6 grid (full resolution)
"""

import os
import glob
import json
from collections import Counter
from typing import List, Optional

from datasets import load_dataset, load_from_disk

from sol_nav.utils.text_builder import (
    build_multires_prompt,
    build_single_res_prompt,
    downsample_frame,
    format_grid_block,
    SYSTEM_PROMPT,
)


# Action space
ACTION_NAMES = {0: "stop", 1: "turn_left", 2: "turn_right", 3: "move_forward"}
NUM_ACTIONS = 4


def _build_episode_frames_multires(
    batch,
    num_grid_r=6,
    num_grid_c=6,
    num_chunk=4,
    long_term_steps=16,
    short_term_steps=2,
    current_steps=1,
    long_term_res=2,
    short_term_res=4,
    current_res=6,
):
    """Expand episodes into per-frame samples with multi-resolution prompts.

    Each sample uses multi-resolution history:
      - long_term_steps frames at long_term_res (downsampled from src_res)
      - short_term_steps frames at short_term_res (downsampled from src_res)
      - current_steps frame at current_res (full resolution)
      - Action chunk: num_chunk future actions

    If not enough history at the beginning, pad by repeating the earliest frame.

    Args:
        batch: dict of lists (batched dataset format).
        num_grid_r, num_grid_c: source grid dimensions (always 6x6).
        num_chunk: number of future action steps to predict.
        long_term_steps: number of long-term history frames.
        short_term_steps: number of short-term history frames.
        current_steps: number of current observation frames.
        long_term_res: grid resolution for long-term history.
        short_term_res: grid resolution for short-term history.
        current_res: grid resolution for current observation.

    Returns:
        dict with keys 'prompt', 'action', 'action_chunk'.
    """
    all_prompts = []
    all_action_chunks = []
    all_actions = []

    # Total context frames needed (history + current)
    total_context = long_term_steps + short_term_steps + current_steps

    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        instr = ep["instruction"]
        frames = ep["frames"]
        total_frames = len(frames)

        for t in range(total_frames):
            # Collect context frames ending at time t
            context_start = max(0, t - total_context + 1)
            actual_context = frames[context_start : t + 1]

            # Pad if insufficient history
            if len(actual_context) < total_context:
                pad_frame = frames[0]
                actual_context = [pad_frame] * (total_context - len(actual_context)) + actual_context

            # Split into long-term, short-term, current
            long_frames = actual_context[:long_term_steps]
            short_frames = actual_context[long_term_steps:long_term_steps + short_term_steps]
            current_frame = actual_context[-1]

            # Build multi-resolution prompt
            prompt = build_multires_prompt(
                instruction=instr,
                current_frame=current_frame,
                short_term_frames=short_frames,
                long_term_frames=long_frames,
                current_res=current_res,
                short_term_res=short_term_res,
                long_term_res=long_term_res,
                src_res=num_grid_r,
            )
            all_prompts.append(prompt)

            # Future action chunk
            chunk = []
            for k in range(num_chunk):
                if t + k < total_frames:
                    chunk.append(frames[t + k]["action"])
                else:
                    chunk.append(0)  # stop as padding
            all_action_chunks.append(chunk)
            all_actions.append(frames[t]["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions,
        "action_chunk": all_action_chunks,
    }


def _build_episode_frames_single_res(
    batch,
    num_grid_r=6,
    num_grid_c=6,
    num_chunk=4,
    num_his=1,
):
    """Expand episodes into per-frame samples with single-resolution prompts.

    Each sample uses `num_his` consecutive frames ending at time t as context.
    If not enough history, pad by repeating the earliest available frame.

    Args:
        batch: dict of lists (batched dataset format).
        num_grid_r, num_grid_c: grid dimensions.
        num_chunk: number of future action steps to predict.
        num_his: number of history frames (including current).

    Returns:
        dict with keys 'prompt', 'action', 'action_chunk'.
    """
    all_prompts = []
    all_action_chunks = []
    all_actions = []

    for i in range(len(batch["episodes"])):
        ep = batch["episodes"][i]
        instr = ep["instruction"]
        frames = ep["frames"]
        total_frames = len(frames)

        for t in range(total_frames):
            # Collect history: [t - num_his + 1, ..., t]
            start_idx = max(0, t - num_his + 1)
            actual_history = frames[start_idx : t + 1]

            # Pad if insufficient history
            if len(actual_history) < num_his:
                pad_frame = frames[start_idx]
                actual_history = [pad_frame] * (num_his - len(actual_history)) + actual_history

            # Build prompt
            prompt = build_single_res_prompt(
                instruction=instr,
                frames=actual_history,
                grid_r=num_grid_r,
                grid_c=num_grid_c,
            )
            all_prompts.append(prompt)

            # Future action chunk
            chunk = []
            for k in range(num_chunk):
                if t + k < total_frames:
                    chunk.append(frames[t + k]["action"])
                else:
                    chunk.append(0)  # stop as padding
            all_action_chunks.append(chunk)
            all_actions.append(frames[t]["action"])

    return {
        "prompt": all_prompts,
        "action": all_actions,
        "action_chunk": all_action_chunks,
    }


def build_solnav_dataset(
    data_dir: str,
    cache_dir: str,
    num_grid_r: int = 6,
    num_grid_c: int = 6,
    num_chunk: int = 4,
    # Multi-resolution config
    long_term_steps: int = 16,
    short_term_steps: int = 2,
    current_steps: int = 1,
    long_term_res: int = 2,
    short_term_res: int = 4,
    current_res: int = 6,
    # Single-resolution fallback
    num_his: int = 1,
    use_multires: bool = True,
    force_rebuild: bool = False,
):
    """Build or load cached SOL-Nav dataset.

    Args:
        data_dir: directory containing merged_part_*.json files.
        cache_dir: path to save/load HuggingFace dataset cache.
        num_grid_r, num_grid_c: source grid dimensions.
        num_chunk: action chunk size.
        long_term_steps: number of long-term history frames.
        short_term_steps: number of short-term history frames.
        current_steps: number of current observation frames.
        long_term_res: grid resolution for long-term history.
        short_term_res: grid resolution for short-term history.
        current_res: grid resolution for current observation.
        num_his: number of history frames for single-res mode.
        use_multires: if True, use multi-resolution prompt builder.
        force_rebuild: if True, ignore cache and rebuild.

    Returns:
        HuggingFace Dataset with columns: prompt, action, action_chunk.
    """
    if not force_rebuild and os.path.exists(cache_dir):
        print(f"Loading cached dataset from {cache_dir}")
        return load_from_disk(cache_dir)

    print("Building SOL-Nav dataset...")
    json_files = sorted(glob.glob(os.path.join(data_dir, "merged_part_*.json")))
    if not json_files:
        raise FileNotFoundError(f"No merged_part_*.json files found in {data_dir}")

    print(f"Found {len(json_files)} data files in {data_dir}")
    raw_ds = load_dataset("json", data_files=json_files, split="train")

    if use_multires:
        print(f"Expanding episodes with multi-resolution grids "
              f"(long={long_term_steps}x{long_term_res}, "
              f"short={short_term_steps}x{short_term_res}, "
              f"current={current_steps}x{current_res}, chunk={num_chunk})...")
        frame_ds = raw_ds.map(
            _build_episode_frames_multires,
            fn_kwargs={
                "num_grid_r": num_grid_r,
                "num_grid_c": num_grid_c,
                "num_chunk": num_chunk,
                "long_term_steps": long_term_steps,
                "short_term_steps": short_term_steps,
                "current_steps": current_steps,
                "long_term_res": long_term_res,
                "short_term_res": short_term_res,
                "current_res": current_res,
            },
            batched=True,
            remove_columns=raw_ds.column_names,
            desc="Building multi-res prompts",
            num_proc=32,
            load_from_cache_file=False,
        )
    else:
        print(f"Expanding episodes with single-resolution grids "
              f"(grid={num_grid_r}x{num_grid_c}, chunk={num_chunk}, his={num_his})...")
        frame_ds = raw_ds.map(
            _build_episode_frames_single_res,
            fn_kwargs={
                "num_grid_r": num_grid_r,
                "num_grid_c": num_grid_c,
                "num_chunk": num_chunk,
                "num_his": num_his,
            },
            batched=True,
            remove_columns=raw_ds.column_names,
            desc="Building single-res prompts",
            num_proc=32,
            load_from_cache_file=False,
        )

    # Statistics
    actions = [ex["action"] for ex in frame_ds]
    print(f"Total frames: {len(frame_ds)}")
    print(f"Action distribution: {Counter(actions)}")

    # Show a sample prompt
    print(f"\n--- Sample prompt (first 800 chars) ---")
    print(frame_ds[0]["prompt"][:800])
    print(f"--- Action chunk: {frame_ds[0]['action_chunk']} ---\n")

    os.makedirs(os.path.dirname(cache_dir) if os.path.dirname(cache_dir) else cache_dir, exist_ok=True)
    frame_ds.save_to_disk(cache_dir)
    print(f"Saved dataset to {cache_dir}")
    return frame_ds


def tokenize_function(examples, tokenizer, max_length=4096):
    """Tokenize prompts for the model.

    Args:
        examples: dict with 'prompt' key (batched).
        tokenizer: HuggingFace tokenizer.
        max_length: maximum token sequence length.

    Returns:
        dict with 'input_ids' and 'attention_mask'.
    """
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
