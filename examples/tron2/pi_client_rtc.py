"""Run TRON2 Real-Time Chunking (RTC) deployment.

Architecture:
  - producer thread: observation -> policy inference -> ActionQueue.merge()
  - consumer thread: ActionQueue.get() -> env.step()
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import logging
import math
import sys
from threading import Event
from threading import Thread
import time
import traceback

from _external_tron2_env import ensure_external_tron2_env_on_path
from deploy_config import PromptController
from deploy_config import age_ms
from deploy_config import bool_value
from deploy_config import build_env_config
from deploy_config import format_obs
from deploy_config import load_deploy_config
from deploy_config import policy_host
from deploy_config import policy_port
from deploy_config import record_paths
from deploy_config import relative_sensor_time_s
from deploy_config import section
from deploy_config import select_profile_path
from deploy_config import timestamp_ms
import numpy as np
from openpi_client import websocket_client_policy
from openpi_client.er2 import ER2Decision
from openpi_client.er2 import ER2Orchestrator
from openpi_client.er2 import ER2PromptMonitor
from openpi_client.er2 import LatestObservationBuffer
from openpi_client.er2 import encode_cam_high_jpeg

ensure_external_tron2_env_on_path()

from tron2_env import Tron2Env
from tron2_env.rtc import ActionQueue
from tron2_env.rtc import LatencyTracker

logger = logging.getLogger(__name__)

DEFAULT_ACTION_HORIZON = 50
ACTION_DIM_RAW = 32
INFERENCE_DELAY_HISTORY_SIZE = 10
TRIGGER_POLL_INTERVAL_S = 0.005
PROCESSED_ARM_INDICES = tuple(range(7)) + tuple(range(8, 15))
PROCESSED_GRIPPER_INDICES = (7, 15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRON2 RTC deployment.")
    parser.add_argument("--profile", type=str, default=None, help="Path to client deployment profile YAML.")
    parser.add_argument(
        "--deploy-config",
        type=str,
        default=None,
        help="Deprecated alias for --profile.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Task prompt sent with each observation. Overrides client.prompt in YAML.",
    )
    return parser.parse_args()


def _safe_int(value, default: int = -1) -> int:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(value):
        return default
    return int(value)


def _p95_int(values) -> int:
    if not values:
        return 0
    ordered = sorted(int(v) for v in values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def _resolve_rtc_timing(client_profile: dict, action_horizon: int) -> tuple[int, int, int]:
    if "execution_horizon" not in client_profile:
        raise ValueError("client.execution_horizon is required for RTC deployment.")
    if "delay" not in client_profile:
        raise ValueError("client.delay is required for RTC deployment.")

    execution_horizon = int(client_profile["execution_horizon"])
    delay = int(client_profile["delay"])
    if not 0 <= execution_horizon <= action_horizon:
        raise ValueError(
            f"client.execution_horizon must satisfy 0 <= s <= H; got s={execution_horizon}, H={action_horizon}."
        )
    if not 0 <= delay < action_horizon:
        raise ValueError(f"client.delay must satisfy 0 <= d < H; got d={delay}, H={action_horizon}.")

    trigger_queue_size = action_horizon - execution_horizon
    return execution_horizon, delay, trigger_queue_size


def _resolve_guidance_weight(client_profile: dict) -> float:
    weight = float(client_profile.get("rtc_guidance_weight", 10.0))
    if weight <= 0:
        raise ValueError(f"client.rtc_guidance_weight must be positive, got {weight}")
    return weight


@dataclass(frozen=True)
class ActionPostprocessConfig:
    enabled: bool = False
    boundary_blend_frames: int = 0
    boundary_blend_curve: str = "smoothstep"
    boundary_blend_scope: str = "arm"
    ema_alpha: float = 1.0
    ema_frames: int = 0
    ema_scope: str = "arm"


class ActionPostProcessor:
    """Optional client-side smoothing for executable processed actions."""

    def __init__(self, config: ActionPostprocessConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled and (
            self.config.boundary_blend_frames > 0 or self.config.ema_alpha < 1.0
        )

    def describe(self) -> str:
        if not self.enabled:
            return "off"
        parts = []
        if self.config.boundary_blend_frames > 0:
            parts.append(
                f"blend={self.config.boundary_blend_frames}:{self.config.boundary_blend_scope}:"
                f"{self.config.boundary_blend_curve}"
            )
        if self.config.ema_alpha < 1.0:
            parts.append(
                f"ema={self.config.ema_alpha:.2f}:{self.config.ema_scope}:"
                f"{self.config.ema_frames or 'all'}"
            )
        return "+".join(parts)

    def apply(
        self,
        processed_actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        merge_delay: int,
    ) -> tuple[np.ndarray, dict[str, float]]:
        diagnostics = {
            "action_postprocess_enabled": float(self.enabled),
            "action_postprocess_merge_delay": float(max(0, merge_delay)),
            "action_postprocess_blend_frames": 0.0,
            "action_postprocess_ema_frames": 0.0,
            "action_postprocess_mae": 0.0,
            "action_postprocess_max": 0.0,
        }
        if not self.enabled:
            return processed_actions, diagnostics

        original = np.asarray(processed_actions)
        actions = original.copy()
        start = max(0, min(int(merge_delay), len(actions)))

        blend_frames = self._apply_boundary_blend(actions, old_processed_leftover, start)
        ema_frames = self._apply_ema(actions, old_processed_leftover, start)

        delta = np.abs(actions.astype(np.float64) - original.astype(np.float64))
        diagnostics.update(
            {
                "action_postprocess_blend_frames": float(blend_frames),
                "action_postprocess_ema_frames": float(ema_frames),
                "action_postprocess_mae": float(np.mean(delta)) if delta.size else 0.0,
                "action_postprocess_max": float(np.max(delta)) if delta.size else 0.0,
            }
        )
        return actions, diagnostics

    def _apply_boundary_blend(
        self,
        actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        start: int,
    ) -> int:
        frames = self.config.boundary_blend_frames
        if frames <= 0 or old_processed_leftover is None or start >= len(actions):
            return 0

        count = min(frames, len(actions) - start, len(old_processed_leftover))
        if count <= 0:
            return 0

        action_dim = min(actions.shape[1], old_processed_leftover.shape[1])
        dims = _processed_scope_indices(self.config.boundary_blend_scope, action_dim)
        if dims.size == 0:
            return 0

        for offset in range(count):
            alpha = (offset + 1) / (count + 1)
            if self.config.boundary_blend_curve == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            actions[start + offset, dims] = (
                (1.0 - alpha) * old_processed_leftover[offset, dims]
                + alpha * actions[start + offset, dims]
            )
        return count

    def _apply_ema(
        self,
        actions: np.ndarray,
        old_processed_leftover: np.ndarray | None,
        start: int,
    ) -> int:
        alpha = self.config.ema_alpha
        if alpha >= 1.0 or start >= len(actions):
            return 0

        count = len(actions) - start
        if self.config.ema_frames > 0:
            count = min(count, self.config.ema_frames)
        if count <= 0:
            return 0

        action_dim = actions.shape[1]
        if old_processed_leftover is not None and len(old_processed_leftover) > 0:
            action_dim = min(action_dim, old_processed_leftover.shape[1])
        dims = _processed_scope_indices(self.config.ema_scope, action_dim)
        if dims.size == 0:
            return 0

        if old_processed_leftover is not None and len(old_processed_leftover) > 0:
            previous = old_processed_leftover[0, dims].astype(np.float64, copy=True)
        else:
            previous = actions[start, dims].astype(np.float64, copy=True)

        for offset in range(count):
            current = actions[start + offset, dims].astype(np.float64, copy=False)
            filtered = alpha * current + (1.0 - alpha) * previous
            actions[start + offset, dims] = filtered
            previous = filtered
        return count


def _resolve_action_postprocess(client_profile: dict) -> ActionPostProcessor:
    raw_config = client_profile.get("rtc_action_postprocess", {}) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("client.rtc_action_postprocess must be a mapping when provided.")

    def option(name: str, legacy_name: str, default):
        return raw_config.get(name, client_profile.get(legacy_name, default))

    config = ActionPostprocessConfig(
        enabled=bool_value(option("enabled", "rtc_action_postprocess_enabled", False)),
        boundary_blend_frames=max(0, int(option("boundary_blend_frames", "rtc_boundary_blend_frames", 0))),
        boundary_blend_curve=str(option("boundary_blend_curve", "rtc_boundary_blend_curve", "smoothstep")),
        boundary_blend_scope=str(option("boundary_blend_scope", "rtc_boundary_blend_scope", "arm")),
        ema_alpha=float(option("ema_alpha", "rtc_action_ema_alpha", 1.0)),
        ema_frames=max(0, int(option("ema_frames", "rtc_action_ema_frames", 0))),
        ema_scope=str(option("ema_scope", "rtc_action_ema_scope", "arm")),
    )

    if config.boundary_blend_curve not in {"linear", "smoothstep"}:
        raise ValueError("client.rtc_action_postprocess.boundary_blend_curve must be 'linear' or 'smoothstep'.")
    for field_name, scope in (
        ("boundary_blend_scope", config.boundary_blend_scope),
        ("ema_scope", config.ema_scope),
    ):
        if scope not in {"arm", "gripper", "all"}:
            raise ValueError(f"client.rtc_action_postprocess.{field_name} must be 'arm', 'gripper', or 'all'.")
    if not 0.0 < config.ema_alpha <= 1.0:
        raise ValueError("client.rtc_action_postprocess.ema_alpha must satisfy 0 < alpha <= 1.")

    return ActionPostProcessor(config)


def _processed_scope_indices(scope: str, action_dim: int) -> np.ndarray:
    if scope == "all":
        return np.arange(action_dim, dtype=np.int64)
    if scope == "arm":
        indices = PROCESSED_ARM_INDICES
    elif scope == "gripper":
        indices = PROCESSED_GRIPPER_INDICES
    else:
        return np.empty((0,), dtype=np.int64)
    return np.asarray([idx for idx in indices if idx < action_dim], dtype=np.int64)


def warmup_rtc(
    ws_client: websocket_client_policy.WebsocketClientPolicy,
    env: Tron2Env,
    prompt_controller: PromptController,
    rtc_guidance_enabled: bool,
    rtc_guidance_weight: float,
    delay: int,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    trained_rtc_mode: bool = False,
):
    """Run warmup inferences before the control loop starts."""
    logger.info("[WARMUP] Starting RTC warmup...")
    obs = env.get_obs()
    obs_formatted = format_obs(obs, prompt=prompt_controller.get())

    t0 = time.perf_counter()
    base_result = ws_client.infer(obs_formatted, return_raw_actions=True)
    t1 = time.perf_counter()
    base_result["client_warmup_latency_s"] = t1 - t0
    logger.info("[WARMUP #1] Non-RTC: %.1fms", (t1 - t0) * 1000)

    dummy_left_over = np.zeros((action_horizon, ACTION_DIM_RAW), dtype=np.float32)
    if trained_rtc_mode:
        rtc_kwargs = {
            "trained_rtc_mode": True,
            "inference_delay": delay,
            "return_raw_actions": True,
            "prev_chunk_left_over": dummy_left_over,
        }
    elif rtc_guidance_enabled:
        rtc_kwargs = {
            "inference_delay": delay,
            "max_guidance_weight": rtc_guidance_weight,
            "prefix_horizon": 0,
            "return_raw_actions": True,
            "prev_chunk_left_over": dummy_left_over,
            "prev_chunk_left_over_len": 0,
        }
    else:
        logger.info("[WARMUP] RTC guidance disabled; replace-only mode uses non-RTC warmup actions.")
        return base_result

    t0 = time.perf_counter()
    rtc_result = ws_client.infer(obs_formatted, **rtc_kwargs)
    t1 = time.perf_counter()
    logger.info(
        "[WARMUP #2] RTC path: %.1fms, raw=%s, proc=%s",
        (t1 - t0) * 1000,
        rtc_result["raw_actions"].shape,
        np.stack(rtc_result["actions"], axis=0).shape,
    )
    logger.info("[WARMUP] Complete.")
    return base_result


def inference_producer(
    ws_client: websocket_client_policy.WebsocketClientPolicy,
    env: Tron2Env,
    action_queue: ActionQueue,
    latency_stats: LatencyTracker,
    shutdown_event: Event,
    fps: float,
    execution_horizon: int,
    delay: int,
    rtc_guidance_enabled: bool,
    rtc_guidance_weight: float,
    trigger_queue_size: int,
    action_postprocessor: ActionPostProcessor,
    *,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    record_states: list | None = None,
    time_origin: float | None = None,
    perf_origin: float | None = None,
    trained_rtc_mode: bool = False,
    prompt_controller: PromptController | None = None,
    obs_timeout_budget_s: float = 5.0,
    observation_buffer: LatestObservationBuffer | None = None,
):
    """Thread: observation -> inference -> merge into ActionQueue."""
    try:
        logger.info("[PRODUCER] Starting inference producer thread")
        time_per_chunk = 1.0 / fps
        infer_count = 0
        initial_delay = max(0, min(int(delay), action_horizon - 1))
        inference_delay_buffer = deque(maxlen=INFERENCE_DELAY_HISTORY_SIZE)

        while not shutdown_event.is_set():
            if action_queue.qsize() > trigger_queue_size:
                time.sleep(min(TRIGGER_POLL_INTERVAL_S, time_per_chunk * 0.25))
                continue

            merge_delay_cap = max(0, action_horizon - 1)
            inference_delay_p95 = _p95_int(inference_delay_buffer)
            inference_delay = min(inference_delay_p95, merge_delay_cap) if inference_delay_buffer else initial_delay

            obs_request_perf = time.perf_counter()
            obs = None
            obs_wait_start = time.perf_counter()
            while not shutdown_event.is_set():
                try:
                    obs = env.get_obs()
                    break
                except TimeoutError as exc:
                    waited = time.perf_counter() - obs_wait_start
                    if waited >= obs_timeout_budget_s:
                        logger.error(
                            "[PRODUCER] get_obs timed out for %.1fs, exceeding %.1fs.",
                            waited,
                            obs_timeout_budget_s,
                        )
                        raise
                    logger.warning(
                        "[PRODUCER] get_obs timeout %.1f/%.1fs; waiting for a fresh observation: %s",
                        waited,
                        obs_timeout_budget_s,
                        exc,
                    )
            if obs is None:
                break

            if observation_buffer is not None:
                observation_buffer.publish(obs)

            obs_receive_perf = time.perf_counter()
            obs_receive_wall = time.time()
            prompt, prompt_version = (
                prompt_controller.snapshot() if prompt_controller is not None else (None, None)
            )
            obs_formatted = format_obs(obs, prompt=prompt)

            metadata = obs.get("metadata", {})
            image_timestamps = metadata.get("image_timestamps_ms", {}) or {}
            obs_receive_time_s = obs_receive_wall - (time_origin or obs_receive_wall)
            obs_receive_perf_s = obs_receive_perf - (perf_origin or obs_receive_perf)
            ref_timestamp_ms = timestamp_ms(
                metadata.get("observation_ref_timestamp_ms", metadata.get("bridge_ref_timestamp_ms"))
            )
            joint_timestamp_ms = timestamp_ms(metadata.get("joint_timestamp_ms"))
            gripper_timestamp_ms = timestamp_ms(metadata.get("gripper_timestamp_ms"))
            image_timestamp_ms = timestamp_ms(metadata.get("image_timestamp_ms"))
            cam_high_timestamp_ms = timestamp_ms(image_timestamps.get("cam_high"))
            cam_left_wrist_timestamp_ms = timestamp_ms(image_timestamps.get("cam_left_wrist"))
            cam_right_wrist_timestamp_ms = timestamp_ms(image_timestamps.get("cam_right_wrist"))
            obs_age_ms = age_ms(obs_receive_wall, ref_timestamp_ms)
            obs_joint_age_ms = age_ms(obs_receive_wall, joint_timestamp_ms)
            obs_gripper_age_ms = age_ms(obs_receive_wall, gripper_timestamp_ms)
            obs_image_age_ms = age_ms(obs_receive_wall, image_timestamp_ms)

            action_index_before, prev_left_over, queue_size_before = action_queue.snapshot_left_over()
            prev_left_over_len = int(prev_left_over.shape[0]) if prev_left_over is not None else 0
            paper_s = max(0, action_horizon - prev_left_over_len)
            prefix_horizon = prev_left_over_len

            if record_states is not None:
                image_values = [
                    value
                    for value in (
                        cam_high_timestamp_ms,
                        cam_left_wrist_timestamp_ms,
                        cam_right_wrist_timestamp_ms,
                    )
                    if not math.isnan(value)
                ]
                record_states.append(
                    {
                        "infer_index": infer_count,
                        "queue_size": queue_size_before,
                        "action_index_before": action_index_before,
                        "initial_delay": initial_delay,
                        "inference_delay_p95": inference_delay_p95,
                        "inference_delay": inference_delay,
                        "action_horizon": action_horizon,
                        "execution_horizon": execution_horizon,
                        "rtc_guidance_enabled": float(rtc_guidance_enabled),
                        "rtc_guidance_weight": rtc_guidance_weight,
                        "action_postprocess_enabled": float(action_postprocessor.enabled),
                        "trigger_queue_size": trigger_queue_size,
                        "paper_s": paper_s,
                        "prefix_horizon": prefix_horizon,
                        "prev_left_over_len": prev_left_over_len,
                        "obs_request_perf_s": obs_request_perf - (perf_origin or obs_request_perf),
                        "obs_receive_time_s": obs_receive_time_s,
                        "obs_receive_perf_s": obs_receive_perf_s,
                        "obs_wait_ms": (obs_receive_perf - obs_request_perf) * 1000.0,
                        "obs_bridge_ref_timestamp_ms": ref_timestamp_ms,
                        "obs_joint_timestamp_ms": joint_timestamp_ms,
                        "obs_gripper_timestamp_ms": gripper_timestamp_ms,
                        "obs_image_timestamp_ms": image_timestamp_ms,
                        "obs_cam_high_timestamp_ms": cam_high_timestamp_ms,
                        "obs_cam_left_wrist_timestamp_ms": cam_left_wrist_timestamp_ms,
                        "obs_cam_right_wrist_timestamp_ms": cam_right_wrist_timestamp_ms,
                        "obs_age_ms": obs_age_ms,
                        "obs_joint_age_ms": obs_joint_age_ms,
                        "obs_gripper_age_ms": obs_gripper_age_ms,
                        "obs_image_age_ms": obs_image_age_ms,
                        "obs_sensor_time_s": relative_sensor_time_s(obs_receive_time_s, obs_age_ms),
                        "obs_joint_sensor_time_s": relative_sensor_time_s(obs_receive_time_s, obs_joint_age_ms),
                        "obs_joint_ref_offset_ms": (
                            joint_timestamp_ms - ref_timestamp_ms
                            if not math.isnan(joint_timestamp_ms) and not math.isnan(ref_timestamp_ms)
                            else float("nan")
                        ),
                        "obs_gripper_ref_offset_ms": (
                            gripper_timestamp_ms - ref_timestamp_ms
                            if not math.isnan(gripper_timestamp_ms) and not math.isnan(ref_timestamp_ms)
                            else float("nan")
                        ),
                        "obs_image_span_ms": max(image_values) - min(image_values) if image_values else float("nan"),
                        "state": obs["state"].copy(),
                    }
                )

            rtc_kwargs = {"return_raw_actions": True}
            if trained_rtc_mode:
                rtc_kwargs["trained_rtc_mode"] = True
                rtc_kwargs["inference_delay"] = inference_delay
            if rtc_guidance_enabled:
                rtc_kwargs.update(
                    {
                        "inference_delay": inference_delay,
                        "max_guidance_weight": rtc_guidance_weight,
                        "prefix_horizon": prefix_horizon,
                    }
                )
            if (rtc_guidance_enabled or trained_rtc_mode) and prev_left_over is not None:
                if prev_left_over.shape[0] < action_horizon:
                    padded = np.zeros((action_horizon, prev_left_over.shape[1]), dtype=prev_left_over.dtype)
                    padded[: prev_left_over.shape[0]] = prev_left_over
                    prev_left_over = padded
                rtc_kwargs["prev_chunk_left_over"] = prev_left_over
                if rtc_guidance_enabled:
                    rtc_kwargs["prev_chunk_left_over_len"] = prev_left_over_len

            t_infer_start = time.perf_counter()
            result = ws_client.infer(obs_formatted, **rtc_kwargs)
            t_infer_end = time.perf_counter()
            if prompt_controller is not None and prompt_controller.snapshot()[1] != prompt_version:
                logger.info("[PROMPT] Discarding inference from an older prompt")
                continue

            original_actions = result["raw_actions"]
            processed_actions = np.stack(result["actions"], axis=0)

            new_latency = t_infer_end - t_infer_start
            measured_inference_delay = (
                0
                if queue_size_before <= 0
                else min(math.ceil(new_latency / time_per_chunk), merge_delay_cap)
            )
            if infer_count > 0:
                inference_delay_buffer.append(measured_inference_delay)
            latency_stats.add(new_latency)

            if action_index_before is None:
                postprocess_delay = measured_inference_delay
            else:
                postprocess_delay = max(0, action_queue.get_action_index() - action_index_before)
            old_processed_leftover = action_queue.get_processed_left_over() if action_postprocessor.enabled else None
            processed_actions, postprocess_diagnostics = action_postprocessor.apply(
                processed_actions,
                old_processed_leftover,
                postprocess_delay,
            )

            def merge_actions(
                original_actions=original_actions,
                processed_actions=processed_actions,
                measured_inference_delay=measured_inference_delay,
                action_index_before=action_index_before,
            ):
                return action_queue.merge(
                    original_actions,
                    processed_actions,
                    measured_inference_delay,
                    action_index_before,
                    extra_delay=0,
                )

            if prompt_controller is None:
                used_delay = merge_actions()
            else:
                accepted, used_delay = prompt_controller.apply_if_current(prompt_version, merge_actions)
                if not accepted:
                    logger.info("[PROMPT] Discarding inference changed before queue merge")
                    continue
            merge_diagnostics = action_queue.get_last_merge_diagnostics()
            if record_states is not None and record_states:
                record_states[-1].update(merge_diagnostics)
                record_states[-1].update(postprocess_diagnostics)
                record_states[-1].update(
                    {
                        "measured_inference_delay": measured_inference_delay,
                        "used_delay": used_delay,
                        "inference_delay_minus_used": inference_delay - used_delay,
                    }
                )
            if rtc_guidance_enabled and used_delay > inference_delay:
                logger.warning(
                    "RTC delay underflow: d=%d, used_delay=%d. Delay follows recent measured p95.",
                    inference_delay,
                    used_delay,
                )

            server_timing = result.get("server_timing", {})
            policy_timing = result.get("policy_timing", {})
            logger.info(
                "[PRODUCER #%d] e2e=%.1fms server=%.1fms policy=%.1fms "
                "s=%d/H=%d guide=%s d=%d p95=%d meas=%d used=%d left=%d queue=%d->%d post=%s",
                infer_count,
                new_latency * 1000,
                server_timing.get("infer_ms", 0),
                policy_timing.get("infer_ms", 0),
                paper_s,
                action_horizon,
                rtc_guidance_enabled,
                inference_delay,
                inference_delay_p95,
                measured_inference_delay,
                used_delay,
                prev_left_over_len,
                queue_size_before,
                action_queue.qsize(),
                action_postprocessor.describe(),
            )
            infer_count += 1

        logger.info("[PRODUCER] Shutting down after %d inferences", infer_count)
    except Exception:
        logger.error("[PRODUCER] Fatal exception:\n%s", traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


def control_consumer(
    env: Tron2Env,
    action_queue: ActionQueue,
    shutdown_event: Event,
    fps: float,
    *,
    record_data: list | None = None,
    time_origin: float | None = None,
    perf_origin: float | None = None,
    recovery_blend_frames: int = 6,
    prompt_controller: PromptController | None = None,
):
    """Thread: ActionQueue -> env.step()."""
    try:
        logger.info("[CONSUMER] Starting control consumer thread")
        action_interval = 1.0 / fps

        last_action = None
        step_count = 0
        stall_count = 0
        current_source_action_index = None
        recovery_blend_total = max(0, int(recovery_blend_frames))
        recovery_blend_remaining = 0
        recovery_hold_action = None

        while not shutdown_event.is_set():
            start_time = time.perf_counter()
            prompt_version = prompt_controller.snapshot()[1] if prompt_controller is not None else None
            queue_size_before = action_queue.qsize()
            source_action_index = current_source_action_index

            action_index_before_get = action_queue.get_action_index()
            action = action_queue.get()
            if (
                action is not None
                and prompt_controller is not None
                and prompt_controller.snapshot()[1] != prompt_version
            ):
                continue
            recovered_stall_count = 0
            if action is not None:
                recovered_stall_count = stall_count
                current_source_action_index = action_index_before_get
                source_action_index = current_source_action_index
            else:
                recovery_blend_remaining = 0
                recovery_hold_action = None
                stall_count += 1
                if stall_count % 50 == 1:
                    logger.warning("[CONSUMER] Queue empty (stalled %d times), step=%d", stall_count, step_count)

            if action is not None:
                if recovered_stall_count > 0 and last_action is not None:
                    recovery_blend_remaining = recovery_blend_total
                    recovery_hold_action = last_action.copy()
                    if recovery_blend_remaining > 0:
                        logger.warning(
                            "[CONSUMER] Recovered after %d empty ticks; blending next %d frames",
                            recovered_stall_count,
                            recovery_blend_total,
                        )

                if recovery_blend_remaining > 0 and recovery_hold_action is not None:
                    blend_index = recovery_blend_total - recovery_blend_remaining + 1
                    alpha = min(1.0, blend_index / max(1, recovery_blend_total))
                    action = ((1.0 - alpha) * recovery_hold_action + alpha * action).astype(action.dtype, copy=False)
                    recovery_blend_remaining -= 1
                    if recovery_blend_remaining == 0:
                        recovery_hold_action = None

                if last_action is not None:
                    arm_action = np.concatenate((action[:7], action[8:15]))
                    last_arm_action = np.concatenate((last_action[:7], last_action[8:15]))
                    error = np.abs(arm_action - last_arm_action)
                    joint_id = int(np.argmax(error))
                    max_diff = float(error[joint_id])
                    if max_diff >= 0.5:
                        logger.warning(
                            "[CONSUMER] Large action jump: joint %d diff=%.4f at step %d",
                            joint_id,
                            max_diff,
                            step_count,
                        )

                action_start_wall = time.time()
                action_start_perf = time.perf_counter()
                if prompt_controller is None:
                    env.step(action)
                else:
                    accepted, _ = prompt_controller.apply_if_current(prompt_version, lambda action=action: env.step(action))
                    if not accepted:
                        continue
                action_end_perf = time.perf_counter()
                action_end_wall = time.time()

                if record_data is not None:
                    record_data.append(
                        {
                            "step_index": step_count,
                            "queue_size": action_queue.qsize(),
                            "queue_size_before": queue_size_before,
                            "source_action_index": (
                                source_action_index if source_action_index is not None else float("nan")
                            ),
                            "action_start_time_s": action_start_wall - (time_origin or action_start_wall),
                            "action_start_perf_s": action_start_perf - (perf_origin or action_start_perf),
                            "action_end_time_s": action_end_wall - (time_origin or action_end_wall),
                            "action_end_perf_s": action_end_perf - (perf_origin or action_end_perf),
                            "step_duration_ms": (action_end_perf - action_start_perf) * 1000.0,
                            "action": action.copy(),
                        }
                    )
                last_action = action
                step_count += 1
                stall_count = 0

            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, action_interval - elapsed - 0.001)
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("[CONSUMER] Shutting down: %d steps, %d stalls", step_count, stall_count)
    except Exception:
        logger.error("[CONSUMER] Fatal exception:\n%s", traceback.format_exc())
        shutdown_event.set()
        sys.exit(1)


def _save_records(config_profile: dict, record_states: list, record_actions: list) -> None:
    action_save_path, state_save_path = record_paths(
        config_profile,
        action_key="rtc_action_output_path",
        state_key="rtc_state_output_path",
        action_suffix="rtc_action_data",
        state_suffix="rtc_state_data",
    )
    state_meta_fields = [
        "infer_index",
        "queue_size",
        "action_index_before",
        "initial_delay",
        "inference_delay_p95",
        "inference_delay",
        "measured_inference_delay",
        "used_delay",
        "inference_delay_minus_used",
        "action_horizon",
        "execution_horizon",
        "rtc_guidance_enabled",
        "rtc_guidance_weight",
        "action_postprocess_enabled",
        "action_postprocess_merge_delay",
        "action_postprocess_blend_frames",
        "action_postprocess_ema_frames",
        "action_postprocess_mae",
        "action_postprocess_max",
        "trigger_queue_size",
        "paper_s",
        "prefix_horizon",
        "prev_left_over_len",
        "obs_request_perf_s",
        "obs_receive_time_s",
        "obs_receive_perf_s",
        "obs_wait_ms",
        "obs_bridge_ref_timestamp_ms",
        "obs_joint_timestamp_ms",
        "obs_gripper_timestamp_ms",
        "obs_image_timestamp_ms",
        "obs_cam_high_timestamp_ms",
        "obs_cam_left_wrist_timestamp_ms",
        "obs_cam_right_wrist_timestamp_ms",
        "obs_age_ms",
        "obs_joint_age_ms",
        "obs_gripper_age_ms",
        "obs_image_age_ms",
        "obs_sensor_time_s",
        "obs_joint_sensor_time_s",
        "obs_joint_ref_offset_ms",
        "obs_gripper_ref_offset_ms",
        "obs_image_span_ms",
        "merge_count",
        "merge_used_delay",
        "merge_extra_delay",
        "merge_pre_qsize",
        "merge_pre_index",
        "boundary_new_index",
        "boundary_proc_plan_mae",
        "boundary_proc_plan_max",
        "boundary_proc_plan_max_dim",
        "boundary_proc_plan_max_delta",
        "boundary_proc_exec_mae",
        "boundary_proc_exec_max",
        "boundary_proc_exec_max_dim",
        "boundary_proc_exec_max_delta",
        "boundary_proc_plan_arm_mae",
        "boundary_proc_plan_arm_max",
        "boundary_proc_plan_arm_max_dim",
        "boundary_proc_plan_arm_max_delta",
        "boundary_proc_exec_arm_mae",
        "boundary_proc_exec_arm_max",
        "boundary_proc_exec_arm_max_dim",
        "boundary_proc_exec_arm_max_delta",
        "boundary_proc_plan_gripper_mae",
        "boundary_proc_plan_gripper_max",
        "boundary_proc_plan_gripper_max_dim",
        "boundary_proc_plan_gripper_max_delta",
        "boundary_proc_exec_gripper_mae",
        "boundary_proc_exec_gripper_max",
        "boundary_proc_exec_gripper_max_dim",
        "boundary_proc_exec_gripper_max_delta",
        "boundary_raw_plan_mae",
        "boundary_raw_plan_max",
        "boundary_raw_plan_max_dim",
        "boundary_raw_plan_max_delta",
    ]
    action_meta_fields = [
        "step_index",
        "queue_size",
        "queue_size_before",
        "source_action_index",
        "action_start_time_s",
        "action_start_perf_s",
        "action_end_time_s",
        "action_end_perf_s",
        "step_duration_ms",
    ]
    state_header = ",".join(
        state_meta_fields
        + [f"L_arm_{i}" for i in range(7)]
        + ["L_grip"]
        + [f"R_arm_{i}" for i in range(7)]
        + ["R_grip"]
    )
    action_header = ",".join(
        action_meta_fields
        + [f"L_arm_{i}" for i in range(7)]
        + ["L_grip"]
        + [f"R_arm_{i}" for i in range(7)]
        + ["R_grip"]
    )

    if record_states:
        state_array = np.vstack(
            [
                np.concatenate(
                    (
                        np.array([record.get(field, float("nan")) for field in state_meta_fields], dtype=np.float64),
                        record["state"],
                    )
                )
                for record in record_states
            ]
        )
        np.savetxt(state_save_path, state_array, delimiter=",", fmt="%.6f", header=state_header, comments="")
        logger.info("Saved %d states to %s (shape=%s)", len(record_states), state_save_path, state_array.shape)
    if record_actions:
        action_array = np.vstack(
            [
                np.concatenate(
                    (
                        np.array([record.get(field, float("nan")) for field in action_meta_fields], dtype=np.float64),
                        record["action"],
                    )
                )
                for record in record_actions
            ]
        )
        np.savetxt(action_save_path, action_array, delimiter=",", fmt="%.6f", header=action_header, comments="")
        logger.info("Saved %d actions to %s (shape=%s)", len(record_actions), action_save_path, action_array.shape)


def main() -> None:
    args = _parse_args()
    profile_path = select_profile_path(args.profile, args.deploy_config)
    config_profile = load_deploy_config(profile_path)
    client_profile = section(config_profile, "client")
    if not bool_value(client_profile.get("rtc_enabled", False)):
        raise ValueError(
            "client.rtc_enabled is false or missing. Use examples/tron2/pi_client.py for synchronous inference."
        )

    log_level = str(client_profile.get("log_level", "INFO")).upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger("tron2_env.rtc.action_queue").setLevel(logging.WARNING)

    env_config = build_env_config(config_profile)
    fps = float(client_profile.get("fps", env_config.fps))
    rtc_guidance_enabled = bool_value(client_profile.get("rtc_guidance_enabled", True))
    rtc_guidance_weight = _resolve_guidance_weight(client_profile)
    trained_rtc_mode = bool_value(client_profile.get("trained_rtc_mode", False))
    obs_timeout_budget_s = float(client_profile.get("obs_timeout_budget_s", 5.0))
    action_postprocessor = _resolve_action_postprocess(client_profile)

    latency_stats = LatencyTracker()
    shutdown_event = Event()
    prompt_ctrl = PromptController(args.prompt or client_profile.get("prompt"))
    er2_profile = config_profile.get("er2") or {}
    er2_enabled = bool_value(er2_profile.get("enabled", False))
    er2_orchestrator: ER2Orchestrator | None = None
    er2_observations: LatestObservationBuffer | None = None
    er2_monitor: ER2PromptMonitor | None = None
    er2_thread: Thread | None = None

    record_states: list = []
    record_actions: list = []

    logger.info("Observation source: %s", env_config.observation_source)
    logger.info("Control backend: %s", env_config.control_backend)

    with Tron2Env(env_config) as env:
        env.reset()
        logger.info("Robot initialized. Connecting to policy server...")

        if er2_enabled:
            if args.prompt or client_profile.get("prompt"):
                raise ValueError("Do not set client.prompt or --prompt when top-level er2.enabled is true")
            er2_orchestrator = ER2Orchestrator.from_config(er2_profile)
            initial_observation = env.get_obs()
            initial_decision = er2_orchestrator.plan(encode_cam_high_jpeg(initial_observation))
            if initial_decision.decision != "set_skill":
                er2_orchestrator.close()
                raise RuntimeError(
                    "ER2 did not select an initial skill: "
                    f"decision={initial_decision.decision}, evidence={initial_decision.evidence}"
                )
            prompt_ctrl.set(er2_orchestrator.prompt_for(initial_decision.skill_id))
            logger.info(
                "[ER2] Initial skill=%s prompt=%r evidence=%s",
                initial_decision.skill_id,
                prompt_ctrl.get(),
                initial_decision.evidence,
            )

        ws_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host(client_profile),
            port=policy_port(client_profile),
        )

        server_meta = ws_client.get_server_metadata()
        rtc_enabled = bool(server_meta.get("rtc_enabled", False))
        if not rtc_enabled:
            raise RuntimeError(
                "The policy server does not report rtc_enabled=True. Use pi_client.py for non-RTC inference "
                "or start a server with an RTC-capable model."
            )
        action_horizon = int(server_meta.get("action_horizon", DEFAULT_ACTION_HORIZON))
        execution_horizon, delay, trigger_queue_size = _resolve_rtc_timing(client_profile, action_horizon)
        logger.info("Server action_horizon=%d", action_horizon)
        logger.info(
            "Config: fps=%.1f, H=%d, s=%d, init_d=%d, trigger_queue_size=H-s=%d, "
            "guide=%s, weight=%.2f, trained_rtc=%s, post=%s",
            fps,
            action_horizon,
            execution_horizon,
            delay,
            trigger_queue_size,
            rtc_guidance_enabled,
            rtc_guidance_weight,
            trained_rtc_mode,
            action_postprocessor.describe(),
        )

        action_queue = ActionQueue(rtc_enabled=rtc_enabled)
        time_origin = time.time()
        perf_origin = time.perf_counter()

        warmup_result = warmup_rtc(
            ws_client,
            env,
            prompt_ctrl,
            rtc_guidance_enabled,
            rtc_guidance_weight,
            delay,
            action_horizon=action_horizon,
            trained_rtc_mode=trained_rtc_mode,
        )

        warmup_raw = warmup_result["raw_actions"]
        warmup_proc = np.stack(warmup_result["actions"], axis=0)
        action_queue.merge(warmup_raw, warmup_proc, real_delay=0)
        prompt_ctrl.set_on_change(action_queue.clear)
        if er2_orchestrator is None:
            prompt_ctrl.start_stdin_listener()
        else:
            logger.info("[ER2] Live stdin prompt switching is disabled while ER2 owns orchestration.")

        if er2_orchestrator is not None:
            er2_observations = LatestObservationBuffer()

            def apply_er2_decision(decision: ER2Decision, orchestrator: ER2Orchestrator) -> None:
                logger.info(
                    "[ER2] decision=%s skill=%s previous_status=%s goal=%s evidence=%s",
                    decision.decision,
                    decision.skill_id,
                    decision.previous_skill_status,
                    decision.goal_status,
                    decision.evidence,
                )
                if decision.decision == "set_skill":
                    prompt_ctrl.set(orchestrator.prompt_for(decision.skill_id))
                elif decision.is_terminal:
                    prompt_ctrl.set(None)

            er2_monitor = ER2PromptMonitor(
                er2_orchestrator,
                er2_observations,
                shutdown_event,
                apply_er2_decision,
                interval_s=float(er2_profile.get("interval_s", 1.0)),
                jpeg_quality=int(er2_profile.get("jpeg_quality", 85)),
                initial_skill_id=initial_decision.skill_id,
            )

        latency_stats.reset()
        warmup_latency_s = float(warmup_result.get("client_warmup_latency_s") or 0.0)
        if warmup_latency_s > 0:
            latency_stats.add(warmup_latency_s)
        logger.info(
            "[INIT] Queue seeded with %d warmup actions, latency seed %.1fms/%d frames",
            action_queue.qsize(),
            warmup_latency_s * 1000.0,
            math.ceil(warmup_latency_s * fps) if warmup_latency_s > 0 else 0,
        )

        producer_thread = Thread(
            target=inference_producer,
            args=(
                ws_client,
                env,
                action_queue,
                latency_stats,
                shutdown_event,
                fps,
                execution_horizon,
                delay,
                rtc_guidance_enabled,
                rtc_guidance_weight,
                trigger_queue_size,
                action_postprocessor,
            ),
            kwargs={
                "action_horizon": action_horizon,
                "record_states": record_states,
                "time_origin": time_origin,
                "perf_origin": perf_origin,
                "trained_rtc_mode": trained_rtc_mode,
                "prompt_controller": prompt_ctrl,
                "obs_timeout_budget_s": obs_timeout_budget_s,
                "observation_buffer": er2_observations,
            },
            daemon=True,
            name="Producer",
        )
        producer_thread.start()

        consumer_thread = Thread(
            target=control_consumer,
            args=(env, action_queue, shutdown_event, fps),
            kwargs={
                "record_data": record_actions,
                "time_origin": time_origin,
                "perf_origin": perf_origin,
                "recovery_blend_frames": int(client_profile.get("obs_recovery_blend_frames", 6)),
                "prompt_controller": prompt_ctrl,
            },
            daemon=True,
            name="Consumer",
        )
        consumer_thread.start()

        if er2_monitor is not None:
            er2_thread = Thread(target=er2_monitor.run, daemon=True, name="ER2Monitor")
            er2_thread.start()

        duration = float(client_profile.get("duration", 120.0))
        if duration and duration > 0:
            logger.info("Running for %.0f seconds...", duration)
        else:
            logger.info("Running indefinitely (duration=0). Ctrl+C to stop.")
        start_time = time.time()

        try:
            while not shutdown_event.is_set() and (duration <= 0 or (time.time() - start_time) < duration):
                time.sleep(5)
                logger.info(
                    "[MAIN] queue=%d, action_idx=%d, latency_p95=%.1fms",
                    action_queue.qsize(),
                    action_queue.get_action_index(),
                    (latency_stats.p95() or 0) * 1000,
                )
        except KeyboardInterrupt:
            logger.info("Interrupted by operator.")
        finally:
            logger.info("Stopping RTC threads")
            shutdown_event.set()
            if er2_thread is not None:
                er2_thread.join(timeout=5)
            producer_thread.join(timeout=5)
            consumer_thread.join(timeout=5)
            _save_records(config_profile, record_states, record_actions)

        if er2_orchestrator is not None:
            er2_orchestrator.close()

    logger.info("Cleanup completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", force=True)
    main()
