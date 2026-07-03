"""
SOL-Nav Structured Observation Prompt Builder.

Converts egocentric RGB-D observations into structured textual descriptions
following the SOL-Nav paper supplement format. Supports multi-resolution grids:
  - Long-term history: 2x2 grid (downsampled from 6x6)
  - Short-term history: 4x4 grid (downsampled from 6x6)
  - Current observation: 6x6 grid (full resolution)

Prompt format (matching supplement material):
  ### System Description:
  You are a robot that ...

  ### Structured Obervation:

  [Time Step -18] Long Observation Grid:
  [0,0]: depth=2.31, semantic=ceiling, color=light_gray; [0,1]: ...
  [1,0]: depth=2.98, semantic=ceiling, color=gray; [1,1]: ...

  ...more long-term frames...

  [Time Step -2] Short Observation Grid:
  [0,0]: depth=2.11, semantic=wall, color=yellow; [0,1]: ...
  ...

  [Time Step 0] Current Observation Grid:
  [0,0]: depth=2.08, semantic=window, color=gray; [0,1]: ...
  ...

  ### Task Instruction:
  <navigation instruction text>
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional


SYSTEM_PROMPT = (
    "### System Description:\n"
    "You are a robot that can turn left or right by a specific degree, "
    "move forward a certain distance, or stop. You must decide your next "
    "action based on the following sequence of time-stamped Observation "
    "Grids and the Task Instruction."
)


def _majority_vote(values: list) -> str:
    """Return the most common value in a list."""
    if not values:
        return "unknown"
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def downsample_grid(
    patches: dict,
    src_r: int,
    src_c: int,
    dst_r: int,
    dst_c: int,
) -> dict:
    """Downsample a grid of patches from (src_r, src_c) to (dst_r, dst_c).

    Each destination cell covers approximately (src_r/dst_r) x (src_c/dst_c)
    source cells. For depth values, the average is taken. For semantic/color,
    majority vote is used.

    Args:
        patches: dict mapping '(i,j)' -> value, at source resolution.
        src_r, src_c: source grid dimensions.
        dst_r, dst_c: destination grid dimensions.

    Returns:
        dict mapping '(i,j)' -> value, at destination resolution.
    """
    result = {}
    for di in range(dst_r):
        for dj in range(dst_c):
            # Compute source cell range
            si_start = int(di * src_r / dst_r)
            si_end = int((di + 1) * src_r / dst_r)
            sj_start = int(dj * src_c / dst_c)
            sj_end = int((dj + 1) * src_c / dst_c)

            values = []
            for si in range(si_start, si_end):
                for sj in range(sj_start, sj_end):
                    key = f"({si},{sj})"
                    if key in patches:
                        values.append(patches[key])

            if not values:
                result[f"({di},{dj})"] = 0.0
            elif isinstance(values[0], (int, float)):
                # Depth: average
                result[f"({di},{dj})"] = sum(values) / len(values)
            else:
                # Semantic / Color: majority vote
                result[f"({di},{dj})"] = _majority_vote(values)

    return result


def downsample_frame(
    frame: dict,
    src_r: int,
    src_c: int,
    dst_r: int,
    dst_c: int,
) -> dict:
    """Downsample all patches in a frame from source to destination resolution.

    Args:
        frame: dict with 'depth_patches', 'semantic_patches', 'color_patches'.
        src_r, src_c: source grid dimensions.
        dst_r, dst_c: destination grid dimensions.

    Returns:
        New frame dict with downsampled patches.
    """
    return {
        "depth_patches": downsample_grid(frame.get("depth_patches", {}), src_r, src_c, dst_r, dst_c),
        "semantic_patches": downsample_grid(frame.get("semantic_patches", {}), src_r, src_c, dst_r, dst_c),
        "color_patches": downsample_grid(frame.get("color_patches", {}), src_r, src_c, dst_r, dst_c),
    }


def format_grid_block(frame: dict, grid_r: int, grid_c: int) -> str:
    """Format a single frame's observation as a grid block.

    Args:
        frame: dict with keys 'depth_patches', 'semantic_patches', 'color_patches',
               each mapping '(i,j)' -> value.
        grid_r, grid_c: grid dimensions.

    Returns:
        Multi-line string like:
            [0,0]: depth=2.31, semantic=ceiling, color=light_gray; [0,1]: ...
            [1,0]: depth=2.98, semantic=ceiling, color=gray; [1,1]: ...
    """
    dp = frame.get("depth_patches", {})
    sp = frame.get("semantic_patches", {})
    cp = frame.get("color_patches", {})

    rows = []
    for i in range(grid_r):
        cells = []
        for j in range(grid_c):
            key = f"({i},{j})"
            d_val = dp.get(key, 0.0)
            s_val = sp.get(key, "unknown")
            c_val = cp.get(key, "unknown")
            cells.append(
                f"[{i},{j}]: depth={d_val:.2f}, semantic={s_val}, color={c_val}"
            )
        rows.append("; ".join(cells))
    return "\n".join(rows)


def build_multires_prompt(
    instruction: str,
    current_frame: dict,
    short_term_frames: List[dict],
    long_term_frames: List[dict],
    current_res: int = 6,
    short_term_res: int = 4,
    long_term_res: int = 2,
    src_res: int = 6,
) -> str:
    """Build the complete SOL-Nav structured observation prompt with multi-resolution.

    Follows the supplement material format:
      1. System Description
      2. Long-term history frames (low-res grid, e.g. 2x2) from oldest to newest
      3. Short-term history frames (medium-res grid, e.g. 4x4) from oldest to newest
      4. Current observation frame (high-res grid, e.g. 6x6)
      5. Task Instruction

    Time step numbering:
      - Long-term: -(n_long + n_short) to -(n_short + 1) (e.g., -18 to -3)
      - Short-term: -n_short to -1 (e.g., -2, -1)
      - Current: 0

    Args:
        instruction: natural language navigation instruction.
        current_frame: dict with patches at source resolution.
        short_term_frames: list of frame dicts for short-term history (oldest first).
        long_term_frames: list of frame dicts for long-term history (oldest first).
        current_res: grid resolution for current observation.
        short_term_res: grid resolution for short-term history.
        long_term_res: grid resolution for long-term history.
        src_res: source grid resolution in the data (always 6).

    Returns:
        Complete prompt string.
    """
    n_long = len(long_term_frames)
    n_short = len(short_term_frames)

    parts = [SYSTEM_PROMPT, "", "### Structured Obervation:", ""]

    # Long-term history frames (oldest to newest)
    # Time steps: -(n_long + n_short) to -(n_short + 1)
    for idx, frame in enumerate(long_term_frames):
        time_step = idx - (n_long + n_short)
        if long_term_res < src_res:
            ds_frame = downsample_frame(frame, src_res, src_res, long_term_res, long_term_res)
        else:
            ds_frame = frame
        block = format_grid_block(ds_frame, long_term_res, long_term_res)
        parts.append(f"[Time Step {time_step}] Long Observation Grid:")
        parts.append(block)
        parts.append("")

    # Short-term history frames (oldest to newest)
    # Time steps: -n_short to -1
    for idx, frame in enumerate(short_term_frames):
        time_step = idx - n_short
        if short_term_res < src_res:
            ds_frame = downsample_frame(frame, src_res, src_res, short_term_res, short_term_res)
        else:
            ds_frame = frame
        block = format_grid_block(ds_frame, short_term_res, short_term_res)
        parts.append(f"[Time Step {time_step}] Short Observation Grid:")
        parts.append(block)
        parts.append("")

    # Current observation (full resolution)
    if current_res < src_res:
        ds_frame = downsample_frame(current_frame, src_res, src_res, current_res, current_res)
    else:
        ds_frame = current_frame
    block = format_grid_block(ds_frame, current_res, current_res)
    parts.append("[Time Step 0] Current Observation Grid:")
    parts.append(block)
    parts.append("")

    # Task instruction
    parts.append("### Task Instruction:")
    parts.append(instruction)

    return "\n".join(parts)


def build_single_res_prompt(
    instruction: str,
    frames: list,
    grid_r: int = 6,
    grid_c: int = 6,
) -> str:
    """Build prompt with single grid resolution and history (backwards compatible).

    Args:
        instruction: navigation instruction.
        frames: list of frame dicts, ordered oldest to newest.
        grid_r, grid_c: grid dimensions.

    Returns:
        Formatted prompt string.
    """
    parts = [SYSTEM_PROMPT, "", "### Structured Obervation:", ""]

    num_his = len(frames)
    for idx, frame in enumerate(frames):
        time_step = idx - (num_his - 1)
        block = format_grid_block(frame, grid_r, grid_c)

        if time_step == 0:
            label = "Current Observation Grid"
        elif time_step >= -2:
            label = "Short Observation Grid"
        else:
            label = "Long Observation Grid"

        parts.append(f"[Time Step {time_step}] {label}:")
        parts.append(block)
        parts.append("")

    parts.append("### Task Instruction:")
    parts.append(instruction)
    return "\n".join(parts)


# Alias for backwards compatibility
build_simple_prompt = build_single_res_prompt


class SOLNavPromptBuilder:
    """Configurable prompt builder for SOL-Nav with multi-resolution support.

    Supports two modes:
    1. Single-resolution mode: all frames use the same grid size (backwards compatible).
    2. Multi-resolution mode: different grid sizes for current/short-term/long-term.
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        data_cfg = config.get("data", {})
        grid_cfg = data_cfg.get("grid_resolutions", {})
        hist_cfg = data_cfg.get("history", {})

        self.current_res = grid_cfg.get("current", 6)
        self.short_term_res = grid_cfg.get("short_term", 4)
        self.long_term_res = grid_cfg.get("long_term", 2)

        self.current_steps = hist_cfg.get("current_steps", 1)
        self.short_term_steps = hist_cfg.get("short_term_steps", 2)
        self.long_term_steps = hist_cfg.get("long_term_steps", 16)

    @property
    def grid_config(self) -> dict:
        return {
            "current": self.current_res,
            "short_term": self.short_term_res,
            "long_term": self.long_term_res,
        }

    @property
    def total_history(self) -> int:
        """Total number of history frames (excluding current)."""
        return self.long_term_steps + self.short_term_steps

    def build_prompt(
        self,
        instruction: str,
        frames: list,
    ) -> str:
        """Build prompt with multi-resolution grids following SOL-Nav paper.

        Args:
            instruction: navigation instruction.
            frames: list of frame dicts, ordered from oldest to newest
                    (last element is current frame).
                    Total length should be long_term_steps + short_term_steps + current_steps.

        Returns:
            Formatted prompt string.
        """
        total = len(frames)
        n_current = self.current_steps
        n_short = self.short_term_steps
        n_long = self.long_term_steps

        # Split frames into resolution groups
        # Current: last n_current frames
        # Short-term: next n_short frames before current
        # Long-term: remaining older frames (up to n_long)
        current_start = max(0, total - n_current)
        short_start = max(0, total - n_current - n_short)
        long_start = max(0, total - n_current - n_short - n_long)

        long_frames = frames[long_start:short_start]
        short_frames = frames[short_start:current_start]
        current_frame = frames[-1] if current_start < total else frames[-1]

        return build_multires_prompt(
            instruction=instruction,
            current_frame=current_frame,
            short_term_frames=short_frames,
            long_term_frames=long_frames,
            current_res=self.current_res,
            short_term_res=self.short_term_res,
            long_term_res=self.long_term_res,
        )

    def build_prompt_single_res(
        self,
        instruction: str,
        frames: list,
        grid_r: int = 6,
        grid_c: int = 6,
    ) -> str:
        """Build prompt with single grid resolution (backwards compatible)."""
        return build_single_res_prompt(instruction, frames, grid_r, grid_c)
