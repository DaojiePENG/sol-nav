"""Logging utilities for SOL-Nav."""

import os
import json
import logging
from datetime import datetime


def setup_logging(log_dir: str = None, level=logging.INFO) -> logging.Logger:
    """Setup structured logging for SOL-Nav.

    Args:
        log_dir: directory to save log files.
        level: logging level.

    Returns:
        Logger instance.
    """
    logger = logging.getLogger("sol_nav")
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(os.path.join(log_dir, f"solnav_{timestamp}.log"))
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def save_sample_outputs(
    samples: list,
    save_path: str,
):
    """Save sample prompts and model outputs for analysis.

    Args:
        samples: list of dicts with keys 'prompt', 'true_actions', 'pred_actions'.
        save_path: path to save JSON file.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(samples)} sample outputs to {save_path}")
