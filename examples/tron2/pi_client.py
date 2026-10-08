"""Run a synchronous TRON2 real-robot OpenPI client."""

from __future__ import annotations

import argparse
import time

from _external_tron2_env import ensure_external_tron2_env_on_path
from deploy_config import PromptController
from deploy_config import bool_value
from deploy_config import build_env_config
from deploy_config import format_obs
from deploy_config import infer_with_timing
from deploy_config import load_deploy_config
from deploy_config import policy_host
from deploy_config import policy_port
from deploy_config import positive_int_or_none
from deploy_config import record_paths
from deploy_config import section
from deploy_config import select_profile_path
import numpy as np
from openpi_client import websocket_client_policy

ensure_external_tron2_env_on_path()

from tron2_env import Tron2Env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRON2 real-robot policy client.")
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


def _save_records(config_profile: dict, actions: list[np.ndarray], states: list[np.ndarray]) -> None:
    if not actions or not states:
        return
    action_path, state_path = record_paths(
        config_profile,
        action_key="action_output_path",
        state_key="state_output_path",
        action_suffix="action_data",
        state_suffix="state_data",
    )
    np.savetxt(action_path, np.vstack(actions), delimiter=",", fmt="%.6f")
    np.savetxt(state_path, np.vstack(states), delimiter=",", fmt="%.6f")
    print(f"saved actions to {action_path}")
    print(f"saved states to {state_path}")


def main() -> None:
    args = _parse_args()
    profile_path = select_profile_path(args.profile, args.deploy_config)
    config_profile = load_deploy_config(profile_path)
    client_profile = section(config_profile, "client")
    if bool_value(client_profile.get("rtc_enabled", False)):
        raise ValueError(
            "client.rtc_enabled is true. Use examples/tron2/pi_client_rtc.py for RTC deployment."
        )

    env_config = build_env_config(config_profile)
    max_steps = positive_int_or_none(
        client_profile.get("max_steps", client_profile.get("max_inferences", 100)),
        field_name="client.max_steps",
    )
    fps = float(client_profile.get("fps", env_config.fps))
    step_period = 1.0 / fps if fps > 0 else 0.0
    save_record = bool_value(client_profile.get("save_record", False))

    print(f"observation_source: {env_config.observation_source}")
    print(f"control_backend: {env_config.control_backend}")
    print(f"publish_rate: {env_config.publish_rate} Hz, fps: {fps} Hz")

    prompt_ctrl = PromptController(args.prompt or client_profile.get("prompt"))
    prompt_ctrl.start_stdin_listener()

    record_state: list[np.ndarray] = []
    record_action: list[np.ndarray] = []

    with Tron2Env(env_config) as env:
        env.reset()

        ws_client_policy = websocket_client_policy.WebsocketClientPolicy(
            host=policy_host(client_profile),
            port=policy_port(client_profile),
        )

        t = 0
        last_action = env.last_action[:14] if env.last_action is not None else None

        while max_steps is None or t < max_steps:
            print("\n\n", "#" * 10, "begin infer", t, "#" * 10)
            obs_request = time.perf_counter()
            obs = env.get_obs()
            obs_wait_ms = (time.perf_counter() - obs_request) * 1000.0
            if save_record:
                record_state.append(obs["state"].copy())

            prompt, prompt_version = prompt_ctrl.snapshot()
            ans, timing = infer_with_timing(
                ws_client_policy,
                format_obs(obs, prompt=prompt),
            )
            if prompt_ctrl.snapshot()[1] != prompt_version:
                print("[PROMPT] Discarding inference from an older prompt")
                continue
            server_timing = ans.get("server_timing", {})
            policy_timing = ans.get("policy_timing", {})
            print(
                "timing: "
                f"obs_wait={obs_wait_ms:.2f}ms, "
                f"ws_total={timing['total_ms']:.2f}ms, "
                f"pack={timing['pack_ms']:.2f}ms, "
                f"send={timing['send_ms']:.2f}ms, "
                f"recv_wait={timing['recv_wait_ms']:.2f}ms, "
                f"unpack={timing['unpack_ms']:.2f}ms, "
                f"payload={timing['payload_kb']:.1f}KB, "
                f"response={timing['response_kb']:.1f}KB, "
                f"server={server_timing.get('infer_ms', 0):.2f}ms, "
                f"policy={policy_timing.get('infer_ms', 0):.2f}ms"
            )

            actions = np.stack(ans["actions"], axis=0)
            print("left start:", actions[0][:8])
            print("right start:", actions[0][8:])
            print("left end:", actions[-1][:8])
            print("right end:", actions[-1][8:])

            if save_record:
                record_action.append(actions)

            for action in actions:
                if prompt_ctrl.snapshot()[1] != prompt_version:
                    print("[PROMPT] Stopping old prompt action chunk")
                    break
                arm_action = np.concatenate((action[:7], action[8:15]))
                if last_action is not None:
                    error = np.abs(arm_action - last_action)
                    joint_id = int(np.argmax(error))
                    max_diff = float(error[joint_id])
                    if max_diff >= 0.5:
                        print(f"joint {joint_id}'s error is {max_diff:.4f}")

                step_t0 = time.perf_counter()
                accepted, _ = prompt_ctrl.apply_if_current(prompt_version, lambda action=action: env.step(action))
                if not accepted:
                    print("[PROMPT] Stopping old prompt action chunk")
                    break
                last_action = arm_action

                sleep_remaining = step_period - (time.perf_counter() - step_t0)
                if sleep_remaining > 0:
                    time.sleep(sleep_remaining)

            t += 1

        if save_record:
            _save_records(config_profile, record_action, record_state)


if __name__ == "__main__":
    main()
