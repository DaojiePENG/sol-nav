import yaml
import os


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "configs", "default.yaml"
        )
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_args(cfg: dict, args: dict) -> dict:
    """Override config with CLI arguments (only non-None values)."""
    flat = flatten_config(cfg)
    for k, v in args.items():
        if v is not None and k in flat:
            flat[k] = v
    return unflatten_config(flat)


def flatten_config(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_config(d, sep="."):
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result
