"""Train a TRON2 task described by an external YAML config."""

import dataclasses
import pathlib
import sys

import tyro

from openpi.training import tron2_task_config

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import train


@dataclasses.dataclass
class Args:
    task_config: str
    exp_name: str | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = True


def main(args: Args) -> None:
    config = tron2_task_config.create_train_config(args.task_config, exp_name=args.exp_name)
    config = dataclasses.replace(
        config,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=args.wandb_enabled,
    )
    train.main(config)


if __name__ == "__main__":
    main(tyro.cli(Args))
