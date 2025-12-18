import argparse
from pathlib import Path
import yaml

from src.utils.seed import seed_everything
from src.utils.logging import create_run_dir
from src.engine.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg.get("seed", 42))

    base_runs_dir = Path(__file__).resolve().parents[1] / cfg["output"]["runs_dir"]
    run_dir = create_run_dir(base_dir=base_runs_dir, cfg=cfg)

    Trainer(cfg=cfg, run_dir=run_dir).fit()


if __name__ == "__main__":
    main()
