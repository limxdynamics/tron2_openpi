"""Load TRON2 training tasks from external YAML files."""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import yaml

import openpi.models.pi0_config as pi0_config
from openpi.training import config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class Tron2TaskConfig:
    name: str
    repo_id: str
    prompt: str | None = None
    weight_loader: str = "gs://openpi-assets/checkpoints/pi05_base/params"
    num_train_steps: int = 20_000
    save_interval: int = 1000
    batch_size: int = 32
    action_horizon: int = 50
    state_dim: int = 16
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    cam_high_key: str = "observation.images.cam_high"
    cam_left_wrist_key: str = "observation.images.cam_left_wrist"
    cam_right_wrist_key: str = "observation.images.cam_right_wrist"
    state_key: str = "observation.state"
    action_key: str = "action"
    prompt_from_task: bool = False
    adapt_to_pi: bool = False
    use_delta_joint_actions: bool = False
    rtc_training_simulated_delay: int | None = None
    crop_config: dict[str, dict[str, int]] = dataclasses.field(
        default_factory=lambda: {
            "observation.images.cam_high": {"x": 120, "y": 140, "w": 500, "h": 340},
            "observation.images.cam_left_wrist": {"x": 0, "y": 0, "w": 640, "h": 480},
            "observation.images.cam_right_wrist": {"x": 0, "y": 0, "w": 640, "h": 480},
        }
    )


def _read_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).expanduser().open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Task config must be a mapping: {path}")
    return data


def _normalize_crop_config(
    crop_config: dict[str, dict[str, int]],
    *,
    cam_high_key: str,
    cam_left_wrist_key: str,
    cam_right_wrist_key: str,
) -> dict[str, dict[str, int]]:
    internal_names = {
        cam_high_key: "cam_high",
        cam_left_wrist_key: "cam_left_wrist",
        cam_right_wrist_key: "cam_right_wrist",
    }
    return {
        internal_names.get(source_key, source_key): box for source_key, box in crop_config.items()
    }


def load_task(path: str | pathlib.Path) -> Tron2TaskConfig:
    data = _read_yaml(path)
    allowed = {field.name for field in dataclasses.fields(Tron2TaskConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown TRON2 task fields in {path}: {', '.join(unknown)}")
    task = Tron2TaskConfig(**data)
    if task.prompt_from_task and task.prompt:
        raise ValueError("prompt must not be set when prompt_from_task is true")
    if not task.prompt_from_task and not task.prompt:
        raise ValueError("prompt is required unless prompt_from_task is true")
    return task


def create_train_config(path: str | pathlib.Path, *, exp_name: str | None = None) -> _config.TrainConfig:
    task = load_task(path)
    repack_structure: dict[str, Any] = {
        "images": {
            "cam_high": task.cam_high_key,
            "cam_left_wrist": task.cam_left_wrist_key,
            "cam_right_wrist": task.cam_right_wrist_key,
        },
        "state": task.state_key,
        "actions": task.action_key,
    }
    if task.prompt_from_task:
        repack_structure["prompt"] = "prompt"

    repack_transforms = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
    return _config.TrainConfig(
        name=task.name,
        exp_name=exp_name or task.name,
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=task.action_horizon,
            rtc_training_simulated_delay=task.rtc_training_simulated_delay,
        ),
        data=_config.LeRobotTronDataConfig(
            repo_id=task.repo_id,
            default_prompt=task.prompt,
            base_config=_config.DataConfig(prompt_from_task=task.prompt_from_task),
            use_delta_joint_actions=task.use_delta_joint_actions,
            adapt_to_pi=task.adapt_to_pi,
            state_dim=task.state_dim,
            crop_config=_normalize_crop_config(
                task.crop_config,
                cam_high_key=task.cam_high_key,
                cam_left_wrist_key=task.cam_left_wrist_key,
                cam_right_wrist_key=task.cam_right_wrist_key,
            ),
            repack_transforms=repack_transforms,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(task.weight_loader),
        num_train_steps=task.num_train_steps,
        save_interval=task.save_interval,
        batch_size=task.batch_size,
        assets_base_dir=task.assets_base_dir,
        checkpoint_base_dir=task.checkpoint_base_dir,
    )
