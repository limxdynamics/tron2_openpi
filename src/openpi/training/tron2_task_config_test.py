import pathlib

import pytest

import openpi.transforms as _transforms
from openpi.training import tron2_task_config


def _write_task(path: pathlib.Path, *, prompt_from_task: bool = False) -> None:
    lines = [
        "name: pi05_tron2_test",
        "repo_id: test_dataset",
        "weight_loader: /tmp/weights/params",
        f"prompt_from_task: {str(prompt_from_task).lower()}",
    ]
    if not prompt_from_task:
        lines.append("prompt: fixed prompt")
    path.write_text("\n".join(lines))


def test_load_task_supports_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    _write_task(task_path, prompt_from_task=True)

    task = tron2_task_config.load_task(task_path)

    assert task.prompt_from_task is True


def test_load_task_rejects_prompt_with_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "\n".join(
            [
                "name: pi05_tron2_test",
                "repo_id: test_dataset",
                "prompt: ambiguous prompt",
                "weight_loader: /tmp/weights/params",
                "prompt_from_task: true",
            ]
        )
    )

    with pytest.raises(ValueError, match="prompt must not be set when prompt_from_task is true"):
        tron2_task_config.load_task(task_path)


def test_load_task_requires_prompt_without_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "\n".join(
            [
                "name: pi05_tron2_test",
                "repo_id: test_dataset",
                "weight_loader: /tmp/weights/params",
            ]
        )
    )

    with pytest.raises(ValueError, match="prompt is required unless prompt_from_task is true"):
        tron2_task_config.load_task(task_path)


def test_load_task_allows_prompt_from_task_without_fallback_prompt(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "\n".join(
            [
                "name: pi05_tron2_test",
                "repo_id: test_dataset",
                "weight_loader: /tmp/weights/params",
                "prompt_from_task: true",
            ]
        )
    )

    task = tron2_task_config.load_task(task_path)

    assert task.prompt is None
    assert task.prompt_from_task is True


def test_create_train_config_propagates_prompt_from_task(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    _write_task(task_path, prompt_from_task=True)

    config = tron2_task_config.create_train_config(task_path)

    assert config.data.base_config is not None
    assert config.data.base_config.prompt_from_task is True


def test_create_train_config_preserves_task_prompt_through_repack(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    _write_task(task_path, prompt_from_task=True)

    config = tron2_task_config.create_train_config(task_path)
    data_config = config.data.create(config.assets_dirs, config.model)
    repack = data_config.repack_transforms.inputs[0]

    repacked = repack(
        {
            "observation.images.cam_high": "high",
            "observation.images.cam_left_wrist": "left",
            "observation.images.cam_right_wrist": "right",
            "observation.state": "state",
            "action": "action",
            "prompt": "episode task prompt",
        }
    )

    assert repacked["prompt"] == "episode task prompt"


def test_create_train_config_propagates_crop_config_for_model_transforms(tmp_path: pathlib.Path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        "\n".join(
            [
                "name: pi05_tron2_test",
                "repo_id: test_dataset",
                "prompt: fixed prompt",
                "weight_loader: /tmp/weights/params",
                "crop_config:",
                "  observation.images.cam_high:",
                "    x: 1",
                "    y: 2",
                "    w: 3",
                "    h: 4",
            ]
        )
    )

    config = tron2_task_config.create_train_config(task_path)
    data_config = config.data.create(config.assets_dirs, config.model)

    assert isinstance(data_config.model_transforms.inputs[0], _transforms.CropImages)
    assert data_config.model_transforms.inputs[0].crop_config["cam_high"] == {"x": 1, "y": 2, "w": 3, "h": 4}
