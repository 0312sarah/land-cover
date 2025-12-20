import argparse
from pathlib import Path
import yaml

from src.utils.seed import seed_everything
from src.utils.logging import create_run_dir
from src.engine.trainer import Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, nargs="+", help="Path(s) to YAML config(s).")
    p.add_argument("--config-dir", type=str, help="Directory containing YAML configs to run sequentially.")
    args = p.parse_args()

    cfg_paths = []
    if args.config:
        cfg_paths.extend(args.config)
    if args.config_dir:
        cfg_dir = Path(args.config_dir)
        cfg_paths.extend(sorted(str(p) for p in cfg_dir.glob("*.yaml")))
    if not cfg_paths:
        raise ValueError("Provide --config (one or many) or --config-dir with YAML files.")

    for cfg_path in cfg_paths:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        seed_everything(cfg.get("seed", 42))
        config_name = Path(cfg_path).stem

        base_runs_dir = Path(__file__).resolve().parents[1] / cfg["output"]["runs_dir"]
        run_dir = create_run_dir(base_dir=base_runs_dir, cfg=cfg)

        print(f"[RUN] Starting training for config: {cfg_path}")
        Trainer(cfg=cfg, run_dir=run_dir, config_name=config_name).fit()


if __name__ == "__main__":
    main()
