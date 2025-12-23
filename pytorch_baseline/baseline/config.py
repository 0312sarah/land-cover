"""
Configuration utilities (YAML -> namespace) for the PyTorch baseline.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


def _dict_to_namespace(d: Dict[str, Any]) -> Namespace:
    ns = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        elif isinstance(v, list):
            setattr(
                ns,
                k,
                [(_dict_to_namespace(x) if isinstance(x, dict) else x) for x in v],
            )
        else:
            setattr(ns, k, v)
    return ns


def load_config(config_path: str) -> Namespace:
    """Load a YAML configuration file into a nested Namespace."""
    path = Path(config_path).expanduser().resolve()
    with path.open("r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)

