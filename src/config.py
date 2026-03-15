# src/config.py
from pathlib import Path
import yaml


def load_config(config_path: str = None) -> dict:
    """
    Load settings.yaml from config/ directory.
    All components read from here — no hardcoded values anywhere.
    """
    if config_path is None:
        # Walk up from src/ to project root, then into config/
        root = Path(__file__).parent.parent
        config_path = root / "config" / "settings.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# Singleton — load once, reuse everywhere
CONFIG = load_config()