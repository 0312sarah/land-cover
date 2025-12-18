from __future__ import annotations

from datetime import datetime
from pathlib import Path
import yaml

def create_run_dir(base_dir : Path, cfg: dict) -> Path : 
    """
    Docstring for create_run_dir
    
    :param base_dir: Description
    :type base_dir: Path
    :param cfg: Description
    :type cfg: dict
    :return: Description
    :rtype: Path
    """

    base_dir = Path(base_dir)
    base_dir.mkdir(parents = True, exist_ok=True)

    run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_dir/run_name
    run_dir.mkdir(parents=True, exist_ok = False)

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "tensorboard").mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return run_dir
