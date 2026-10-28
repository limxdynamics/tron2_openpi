import dataclasses
import enum
import inspect
import logging
import os
import socket
import time

import jax
import numpy as np
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import tron2_policy
from openpi.serving import websocket_policy_server
from openpi.shared import deploy_config as _deploy_config
from openpi.training import config as _config

_JAX_CACHE_DIR = os.environ.get("OPENPI_JAX_CACHE_DIR", "/tmp/openpi_jax_cache")
if _JAX_CACHE_DIR:
    jax.config.update("jax_compilation_cache_dir", _JAX_CACHE_DIR)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    TRON2_REAL = "tron2_real"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Preferred YAML profile for local deployment values.
    profile: str | None = None

    # Deprecated alias for --profile.
    deploy_config: str | None = None

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.TRON2_REAL

    # Used when the observation does not include a prompt.
    default_prompt: str | None = None

    # Host and port to serve the policy on.
    host: str | None = None
    port: int | None = None

    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the YAML/default policy is used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # Deprecated compatibility flags. RTC support is now detected from the model.
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "exp"


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.TRON2_REAL: Checkpoint(
        config="pi05_tron2_example",
        dir="checkpoints/pi05_tron2_example/example_checkpoint/19999",
    ),
}


def _with_repo_id(train_config: _config.TrainConfig, repo_id: str | None) -> _config.TrainConfig:
    if repo_id is None:
        return train_config
    return dataclasses.replace(train_config, data=dataclasses.replace(train_config.data, repo_id=repo_id))


def _with_action_horizon(
    train_config: _config.TrainConfig,
    action_horizon: int | None,
) -> _config.TrainConfig:
    if action_horizon is None:
        return train_config
    action_horizon = int(action_horizon)
    if action_horizon <= 0:
        raise ValueError(f"policy.action_horizon must be positive, got {action_horizon}")
    if not hasattr(train_config.model, "action_horizon"):
        raise ValueError(f"Model config {type(train_config.model).__name__} does not support action_horizon")

    model_config = dataclasses.replace(train_config.model, action_horizon=action_horizon)
    logging.info("Overriding inference action_horizon to %d from deploy config", action_horizon)
    return dataclasses.replace(train_config, model=model_config)


def _with_state_dim(train_config: _config.TrainConfig, state_dim: int | None) -> _config.TrainConfig:
    if state_dim is None:
        return train_config
    state_dim = int(state_dim)
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    if not hasattr(train_config.data, "state_dim") or not hasattr(train_config.data, "action_dim"):
        logging.warning("Config %s does not support state_dim; ignoring override.", train_config.name)
        return train_config

    tron2_policy.validate_tron2_dimensions(state_dim, state_dim)
    data_config = dataclasses.replace(train_config.data, state_dim=state_dim, action_dim=state_dim)
    policy_metadata = dict(train_config.policy_metadata or {})
    policy_metadata["state_dim"] = state_dim
    policy_metadata["action_dim"] = state_dim
    logging.info("Overriding TRON2 state/action output dim to %d from deploy config", state_dim)
    return dataclasses.replace(train_config, data=data_config, policy_metadata=policy_metadata)


def _with_delta_actions(train_config: _config.TrainConfig, use_delta: bool | None) -> _config.TrainConfig:
    if use_delta is None:
        return train_config
    if not hasattr(train_config.data, "use_delta_joint_actions"):
        logging.warning("Config %s does not support use_delta_joint_actions; ignoring override.", train_config.name)
        return train_config
    return dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, use_delta_joint_actions=use_delta),
    )


def _get_train_config(
    config_name: str,
    *,
    repo_id: str | None = None,
    action_horizon: int | None = None,
    state_dim: int | None = None,
    use_delta_joint_actions: bool | None = None,
) -> _config.TrainConfig:
    train_config = _config.get_config(config_name)
    train_config = _with_repo_id(train_config, repo_id)
    train_config = _with_action_horizon(train_config, action_horizon)
    train_config = _with_state_dim(train_config, state_dim)
    return _with_delta_actions(train_config, use_delta_joint_actions)


def _create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: str,
    *,
    default_prompt: str | None,
) -> _policy.Policy:
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=default_prompt,
    )
    if hasattr(train_config.data, "state_dim") and hasattr(train_config.data, "action_dim"):
        policy.metadata.setdefault("state_dim", int(train_config.data.state_dim))
        policy.metadata.setdefault("action_dim", int(train_config.data.action_dim))
    return policy


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _create_trained_policy(
            _get_train_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def _policy_bool(policy_profile: dict, key: str) -> bool | None:
    if key not in policy_profile:
        return None
    return _deploy_config.bool_value(policy_profile[key])


def create_policy(args: Args, config_profile: dict) -> _policy.Policy:
    """Create a policy from CLI/YAML arguments."""
    policy_profile = _deploy_config.section(config_profile, "policy")
    client_profile = _deploy_config.section(config_profile, "client")
    default_prompt = args.default_prompt
    if default_prompt is None:
        default_prompt = policy_profile.get("default_prompt")

    repo_id = policy_profile.get("repo_id")
    action_horizon = policy_profile.get("action_horizon")
    state_dim = policy_profile.get("state_dim", client_profile.get("state_dim"))
    use_delta_joint_actions = _policy_bool(policy_profile, "use_delta_joint_actions")

    match args.policy:
        case Checkpoint():
            train_config = _get_train_config(
                args.policy.config,
                repo_id=repo_id,
                action_horizon=action_horizon,
                state_dim=state_dim,
                use_delta_joint_actions=use_delta_joint_actions,
            )
            return _create_trained_policy(
                train_config,
                args.policy.dir,
                default_prompt=default_prompt,
            )
        case Default():
            config_name = policy_profile.get("config")
            checkpoint_dir = policy_profile.get("checkpoint_dir") or policy_profile.get("dir")
            if config_name or checkpoint_dir:
                if not config_name or not checkpoint_dir:
                    raise ValueError("YAML policy config requires both policy.config and policy.checkpoint_dir")
                train_config = _get_train_config(
                    str(config_name),
                    repo_id=repo_id,
                    action_horizon=action_horizon,
                    state_dim=state_dim,
                    use_delta_joint_actions=use_delta_joint_actions,
                )
                return _create_trained_policy(
                    train_config,
                    str(checkpoint_dir),
                    default_prompt=default_prompt,
                )
            return create_default_policy(args.env, default_prompt=default_prompt)


def _policy_supports_rtc(policy: _policy.Policy) -> bool:
    model = getattr(policy, "_model", None)
    sample_actions = getattr(model, "sample_actions", None)
    if sample_actions is None:
        return False
    params = inspect.signature(sample_actions).parameters
    return {
        "inference_delay",
        "prev_chunk_left_over",
        "prev_chunk_left_over_len",
        "prefix_horizon",
        "max_guidance_weight",
    } <= set(params)


def warmup_policy(policy: _policy.Policy, *, rtc_supported: bool) -> None:
    """Trigger JIT compilation before the first real client request."""
    model = getattr(policy, "_model", None)
    action_horizon = int(getattr(model, "action_horizon", 50))
    action_dim = int(getattr(model, "action_dim", 32))
    state_dim = int(policy.metadata.get("state_dim", 16))

    dummy_obs = {
        "state": np.zeros(state_dim, dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
    }

    logging.info("[warmup] Compiling standard inference path...")
    t0 = time.monotonic()
    try:
        policy.infer(dummy_obs)
        logging.info("[warmup] Standard inference path complete in %.1fs", time.monotonic() - t0)
    except Exception:
        logging.exception("[warmup] Standard inference path failed; continuing server startup")

    if rtc_supported:
        dummy_chunk = np.zeros((action_horizon, action_dim), dtype=np.float32)
        logging.info("[warmup] Compiling trained-RTC inference path...")
        t0 = time.monotonic()
        try:
            policy.infer(
                dummy_obs,
                inference_delay=10,
                prev_chunk_left_over=dummy_chunk,
                trained_rtc_mode=True,
            )
            logging.info("[warmup] trained-RTC path complete in %.1fs", time.monotonic() - t0)
        except Exception:
            logging.exception("[warmup] trained-RTC path failed; continuing server startup")


def main(args: Args) -> None:
    profile_path = _deploy_config.select_profile_path(args.profile, args.deploy_config)
    config_profile = _deploy_config.load_deploy_config(profile_path)
    server_profile = _deploy_config.section(config_profile, "server")
    policy_profile = _deploy_config.section(config_profile, "policy")

    policy = create_policy(args, config_profile)
    policy_metadata = policy.metadata

    model = getattr(policy, "_model", None)
    action_horizon = getattr(model, "action_horizon", None)
    if action_horizon is not None:
        policy_metadata["action_horizon"] = int(action_horizon)

    rtc_supported = _policy_supports_rtc(policy)
    policy_metadata["rtc_enabled"] = rtc_supported
    if rtc_supported:
        logging.info("RTC supported; client supplies execution_horizon, delay, and guidance weight.")
    else:
        logging.info("RTC not supported by this model.")
    if (args.rtc_enabled or _deploy_config.bool_value(server_profile.get("rtc_enabled", False))) and not rtc_supported:
        logging.warning("RTC was requested, but this model does not expose RTC parameters.")

    warmup_policy(policy, rtc_supported=rtc_supported)

    record = args.record or _deploy_config.bool_value(policy_profile.get("record", False))
    if record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    host = args.host or str(server_profile.get("host", "0.0.0.0"))
    port = args.port if args.port is not None else int(server_profile.get("port", 8000))

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
