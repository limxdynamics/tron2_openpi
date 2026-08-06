"""Public TRON2 deployment YAML helpers.

This module keeps the example clients small and reads the client-side deploy
profile schema for robot, camera, bridge, and client options.
"""

from __future__ import annotations

from datetime import datetime
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any

from _external_tron2_env import ensure_external_tron2_env_on_path
import einops
import numpy as np
from openpi_client import image_tools

from openpi.shared import deploy_config as _deploy_config

ensure_external_tron2_env_on_path()

from tron2_env import BridgeConfig
from tron2_env import CameraConfig
from tron2_env import EnvConfig
from tron2_env import Tron2Config

logger = logging.getLogger(__name__)

DEFAULT_INIT_JOINTS = [
    0.026899,
    0.2612,
    -0.02709991,
    -1.5477003,
    0.265,
    0.0180999,
    -0.0614999,
    0.008999,
    -0.269,
    0.02069998,
    -1.5567001,
    -0.254,
    -0.02309972,
    0.06469989,
]

OPENPI_CAMERA_NAMES = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
LEGACY_CAMERA_NAME_MAP = {
    "head_camera_image": "cam_high",
    "left_wrist_image": "cam_left_wrist",
    "right_wrist_image": "cam_right_wrist",
}

TRON2_CROP_CONFIG: dict[str, dict[str, int]] = {
    "cam_high": {"x": 120, "y": 140, "w": 500, "h": 340},
    "cam_left_wrist": {"x": 0, "y": 0, "w": 640, "h": 480},
    "cam_right_wrist": {"x": 0, "y": 0, "w": 640, "h": 480},
}


def load_deploy_config(path: str | Path | None) -> dict[str, Any]:
    """Load a public nested deploy profile YAML."""
    return _deploy_config.load_deploy_config(path)


def load_deploy_profile(path: str | Path | None) -> dict[str, Any]:
    """Alias for callers that use the newer profile terminology."""
    return load_deploy_config(path)


def select_profile_path(
    profile: str | Path | None,
    deploy_config: str | Path | None = None,
) -> str | Path | None:
    return _deploy_config.select_profile_path(profile, deploy_config)


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    return _deploy_config.section(config, name)


def bool_value(value: Any) -> bool:
    return _deploy_config.bool_value(value)


def _camera_name(name: str) -> str:
    return LEGACY_CAMERA_NAME_MAP.get(str(name), str(name))


def normalize_camera_profile(camera_profile: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy image names to the OpenPI/TRON2 camera names."""
    profile = dict(camera_profile)
    serial_to_name = profile.get("serial_to_name")
    if isinstance(serial_to_name, dict):
        profile["serial_to_name"] = {
            str(serial): _camera_name(name) for serial, name in serial_to_name.items()
        }

    camera_names = profile.get("camera_names")
    if camera_names:
        profile["camera_names"] = [_camera_name(name) for name in camera_names]
    elif isinstance(serial_to_name, dict):
        profile["camera_names"] = list(profile["serial_to_name"].values())
    else:
        profile["camera_names"] = list(OPENPI_CAMERA_NAMES)

    return profile


def normalized_raw_config(config_profile: dict[str, Any]) -> dict[str, Any]:
    raw_config = dict(config_profile)
    raw_config["camera"] = normalize_camera_profile(section(config_profile, "camera"))
    return raw_config


def build_robot_config(config_profile: dict[str, Any]) -> Tron2Config:
    robot_profile = section(config_profile, "robot")
    init_joints = robot_profile.get("init_joints") or DEFAULT_INIT_JOINTS

    return Tron2Config(
        robot_ip=str(robot_profile.get("ip", "ROBOT_IP")),
        port=int(robot_profile.get("port", 5000)),
        init_joints=init_joints,
        init_head=robot_profile.get("init_head"),
        state_queue_maxlen=int(robot_profile.get("state_queue_maxlen", 7)),
        polling_rate=float(robot_profile.get("polling_rate", 200.0)),
        connection_timeout=float(robot_profile.get("connection_timeout", 5.0)),
    )


def build_camera_config(config_profile: dict[str, Any]) -> CameraConfig:
    camera_profile = normalize_camera_profile(section(config_profile, "camera"))
    resolution = camera_profile.get("resolution", [480, 640, 3])

    return CameraConfig(
        camera_names=list(camera_profile.get("camera_names", OPENPI_CAMERA_NAMES)),
        resolution=tuple(resolution),
        max_queue_size=int(camera_profile.get("max_queue_size", 10)),
        save_debug_images=bool_value(camera_profile.get("save_debug_images", False)),
        debug_image_dir=str(camera_profile.get("debug_image_dir", "debug_images")),
    )


def build_bridge_config(config_profile: dict[str, Any]) -> BridgeConfig:
    bridge_profile = section(config_profile, "bridge")

    return BridgeConfig(
        host=str(bridge_profile.get("host", "wss://BRIDGE_HOST")),
        ws_path=str(bridge_profile.get("ws_path", "/bridge/ws")),
        image_max_fps=int(bridge_profile.get("image_max_fps", 0)),
        align_max_delay_ms=int(bridge_profile.get("align_max_delay_ms", 200)),
        verify_tls=bool_value(bridge_profile.get("verify_tls", False)),
        image_topics=dict(bridge_profile.get("image_topics", BridgeConfig().image_topics)),
        joint_topics=dict(bridge_profile.get("joint_topics", BridgeConfig().joint_topics)),
        save_debug_images=bool_value(bridge_profile.get("save_debug_images", False)),
        debug_image_dir=str(bridge_profile.get("debug_image_dir", "debug_images")),
    )


def build_env_config(config_profile: dict[str, Any]) -> EnvConfig:
    client_profile = section(config_profile, "client")
    robot_profile = section(config_profile, "robot")
    bridge_profile = section(config_profile, "bridge")

    fps = float(client_profile.get("fps", 30.0))
    return EnvConfig(
        robot_config=build_robot_config(config_profile),
        camera_config=build_camera_config(config_profile),
        control_backend=str(
            client_profile.get("control_backend", robot_profile.get("control_backend", "websocket"))
        ),
        publish_rate=float(client_profile.get("publish_rate", robot_profile.get("publish_rate", 300.0))),
        fps=fps,
        time_sync_tolerance=float(client_profile.get("time_sync_tolerance", 0.01)),
        time_sync_max_retries=int(client_profile.get("time_sync_max_retries", 3)),
        legacy_use_time_sync=bool_value(client_profile.get("legacy_use_time_sync", True)),
        state_dim=int(client_profile.get("state_dim", 16)),
        observation_source=str(client_profile.get("observation_source", "legacy")),
        bridge_state_source=str(
            bridge_profile.get("state_source", client_profile.get("bridge_state_source", "bridge"))
        ),
        bridge_config=build_bridge_config(config_profile),
        raw_config=normalized_raw_config(config_profile),
    )


def policy_host(client_profile: dict[str, Any]) -> str:
    return str(client_profile.get("policy_host", client_profile.get("server_host", "127.0.0.1")))


def policy_port(client_profile: dict[str, Any]) -> int:
    return int(client_profile.get("policy_port", client_profile.get("server_port", client_profile.get("port", 8000))))


def positive_int_or_none(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "unlimited"}:
        return None
    steps = int(value)
    if steps <= 0:
        raise ValueError(f"{field_name} must be positive, null, or omitted")
    return steps


def task_name(config_profile: dict[str, Any]) -> str:
    client_profile = section(config_profile, "client")
    policy_profile = section(config_profile, "policy")
    if client_profile.get("task"):
        return str(client_profile["task"])
    config = str(policy_profile.get("config") or "tron2")
    prefix = "pi05_tron2_"
    return config[len(prefix) :] if config.startswith(prefix) else config


def record_paths(
    config_profile: dict[str, Any],
    *,
    action_key: str,
    state_key: str,
    action_suffix: str,
    state_suffix: str,
) -> tuple[Path, Path]:
    """Resolve timestamped action/state CSV paths under client config."""
    client_profile = section(config_profile, "client")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = task_name(config_profile)
    default_action_path = Path("debug_images") / f"{task}_{timestamp}_{action_suffix}.csv"
    default_state_path = Path("debug_images") / f"{task}_{timestamp}_{state_suffix}.csv"
    action_path = Path(client_profile.get(action_key) or default_action_path)
    state_path = Path(client_profile.get(state_key) or default_state_path)
    action_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    return action_path, state_path


def format_obs(obs: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
    """Format a TRON2 observation for the OpenPI websocket server."""
    formatted = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in obs.items()
        if k != "metadata"
    }
    formatted["images"] = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in obs.get("images", {}).items()
    }
    for cam_name, image in formatted["images"].items():
        box = TRON2_CROP_CONFIG.get(cam_name)
        if box is not None:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            image = image[y : y + h, x : x + w]
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
        formatted["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")
    if prompt is not None:
        formatted["prompt"] = prompt
    return formatted


def timestamp_ms(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return timestamp if timestamp > 0 else float("nan")


def age_ms(receive_wall_s: float, timestamp_ms_value: float) -> float:
    if math.isnan(timestamp_ms_value):
        return float("nan")
    return receive_wall_s * 1000.0 - timestamp_ms_value


def relative_sensor_time_s(receive_rel_s: float, age_ms_value: float) -> float:
    if math.isnan(age_ms_value):
        return float("nan")
    return receive_rel_s - age_ms_value / 1000.0


def _clean_prompt(prompt: str | None) -> str | None:
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return None


class PromptController:
    """Thread-safe task prompt holder with optional live stdin updates."""

    def __init__(self, initial: str | None = None):
        self._lock = threading.Lock()
        self._prompt = _clean_prompt(initial)
        self._thread: threading.Thread | None = None

    def get(self) -> str | None:
        with self._lock:
            return self._prompt

    def set(self, prompt: str | None) -> None:
        with self._lock:
            self._prompt = _clean_prompt(prompt)

    def start_stdin_listener(self) -> None:
        if self._thread is not None:
            return
        if not sys.stdin or not sys.stdin.isatty():
            logger.info("Live prompt input disabled; using prompt=%r.", self._prompt)
            return
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="PromptInput")
        self._thread.start()
        logger.info("Live prompt input enabled. Type a new prompt then Enter to switch.")

    def _reader_loop(self) -> None:
        try:
            for line in sys.stdin:
                text = line.strip()
                if not text:
                    continue
                self.set(text)
                logger.info("[PROMPT] Updated task prompt -> %r", self.get())
        except Exception:
            logger.exception("Prompt stdin listener stopped.")


def infer_with_timing(policy, obs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    """Run one websocket inference and return client-side transport timing."""
    from openpi_client import msgpack_numpy

    t0 = time.perf_counter()
    data = policy._packer.pack(obs)
    t1 = time.perf_counter()
    policy._ws.send(data)
    t2 = time.perf_counter()
    response = policy._ws.recv()
    t3 = time.perf_counter()
    if isinstance(response, str):
        raise RuntimeError(f"Error in inference server:\n{response}")
    ans = msgpack_numpy.unpackb(response)
    t4 = time.perf_counter()
    return ans, {
        "pack_ms": (t1 - t0) * 1000.0,
        "send_ms": (t2 - t1) * 1000.0,
        "recv_wait_ms": (t3 - t2) * 1000.0,
        "unpack_ms": (t4 - t3) * 1000.0,
        "total_ms": (t4 - t0) * 1000.0,
        "payload_kb": len(data) / 1024.0,
        "response_kb": len(response) / 1024.0,
    }
