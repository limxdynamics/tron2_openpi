# English | [中文](README_CN.md)

# TRON2 OpenPI

[Installation](INSTALL.md)

`tron2_openpi` is a TRON2 deployment-focused derivative of the OpenPI project.
It keeps the OpenPI policy-serving and pi0/pi0.5 model stack, then adds TRON2
policy transforms, deployment configuration templates, and real-robot client
examples for TRON2.

This repository is meant to be used together with the sibling `tron2_env`
runtime package. It is an integration and deployment example, not a complete
release of private checkpoints, datasets, low-level robot SDKs, or local
deployment profiles.

## What This Repository Provides

- pi0.5 policy serving through `scripts/serve_policy.py`.
- TRON2 policy input/output transforms in `src/openpi/policies/tron2_policy.py`.
- TRON2 training/deployment config registrations in `src/openpi/training/config.py`.
- TRON2 robot clients in `examples/tron2/`.
- Public deployment templates in `configs/deploy/`.
- YAML-driven TRON2 task training via `scripts/train_tron2_task.py` and
  `configs/train/tron2_tasks/example.yaml`.
- Optional bridge observation mode for images and state from TRON2 Bridge.
- Optional legacy RealSense observation mode for directly attached cameras.
- RTC deployment client with warmup, observation-timeout recovery, queue
  diagnostics, and optional action smoothing.
- Modular TRON2 deployment for ServoJ/ServoP arms, gripper/BrainCo2 paired
  hands, head, chassis, and lifter modules.
- OpenPI client package under `packages/openpi-client/`.

## What Is Not Included

- Model weights and checkpoint directories.
- Training datasets, evaluation datasets, logs, or benchmark results.
- Private `.local.yaml` deployment files.
- Credentials, real camera serial numbers, customer data, or private local
  deployment profiles.
- The undeveloped low-level robot transport.
- A safety certification for unattended robot operation.

## Example Tasks and Public Resources

Public task resources are listed below and will be updated as more examples are
released.

| Task | User guide | Model weights | Public deploy profiles |
| --- | --- | --- | --- |
| Candy | [TRON2 OpenPI Candy user guide](https://cwjgfm21di.feishu.cn/wiki/NA5Rw1dWPiu6dwkFAfTcnaFLnQf) | [Hugging Face](https://huggingface.co/limx-tron2/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | `configs/deploy/candy_server.yaml`, `configs/deploy/candy_client.yaml` |
| Cloth | [TRON2 OpenPI Cloth user guide](https://cwjgfm21di.feishu.cn/wiki/AsuRwJQ94igFPvkcyatctA4JnTc) | [Hugging Face](https://huggingface.co/limx-tron2/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | `configs/deploy/cloth_server.yaml`, `configs/deploy/cloth_client.yaml` |
| Sort | [TRON2 OpenPI Sort user guide](https://cwjgfm21di.feishu.cn/wiki/Lm60wXBBligMKHkwDcjcZ1Rlnpb) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | `configs/deploy/sort_client.yaml`, `configs/deploy/sort_server.yaml` |
| Organize the Desktop | [TRON2 OpenPI Organize the Desktop user guide](https://cwjgfm21di.feishu.cn/wiki/COE0whc3giulI6kN5V3cNmj1nxg) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | — |
| Bean Pouring | [TRON2 OpenPI Bean Pouring user guide](https://cwjgfm21di.feishu.cn/wiki/Jm0CwcUXHiz9HMkKkIvcvavhnQb) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | — |
| Clean the Floor | [TRON2 OpenPI Clean the Floor user guide](https://cwjgfm21di.feishu.cn/wiki/CZMRwbgR5ipaDKknTZwcuMibnRg) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limx-tron2/tron2-openpi-models) | — |

Model weights and checkpoints are not stored in this repository. Download the
weights from the linked model repository, then set the actual checkpoint path in
the task server profile before running a robot.

## Repository Layout

```text
parent-directory/
├── tron2_openpi/
│   ├── configs/
│   │   ├── deploy/
│   │   │   ├── candy_server.yaml
│   │   │   ├── candy_client.yaml
│   │   │   ├── tron2_deploy.server.example.yaml
│   │   │   ├── tron2_deploy.client.example.yaml
│   │   │   ├── tron2_deploy.server.example_CN.yaml
│   │   │   └── tron2_deploy.client.example_CN.yaml
│   │   └── train/
│   │       └── tron2_tasks/
│   │           └── example.yaml
│   ├── examples/
│   │   └── tron2/
│   │       ├── deploy_config.py
│   │       ├── pi_client.py
│   │       └── pi_client_rtc.py
│   ├── packages/
│   │   └── openpi-client/
│   ├── scripts/
│   │   ├── cloud_train_entrypoint_portable.sh
│   │   ├── compute_norm_stats.py
│   │   ├── serve_policy.py
│   │   └── train_tron2_task.py
│   └── src/
│       └── openpi/
└── tron2_env/
```

Keep `tron2_openpi/` and `tron2_env/` side by side. The TRON2 client adds the
sibling `../tron2_env/src` path at startup so it can import the runtime package.
Recorded-action replay utilities live in `../tron2_env/examples/replay_data.py`.

## Installation

Environment requirements and setup commands live in [INSTALL.md](INSTALL.md).
For the Chinese installation guide, see [INSTALL_CN.md](INSTALL_CN.md).

## Deployment

### Deployment Configuration

Deployment uses two profiles per task:

- Server profile: model checkpoint, prompt, policy overrides, and server port.
- Client profile: policy-server address, robot endpoint, observation source,
  camera/Bridge settings, execution loop, and RTC settings.

Use the public Candy profiles or generic templates as starting points:

- Candy server profile: `configs/deploy/candy_server.yaml`
- Candy client profile: `configs/deploy/candy_client.yaml`
- English templates: `configs/deploy/tron2_deploy.server.example.yaml`,
  `configs/deploy/tron2_deploy.client.example.yaml`
- Chinese templates: `configs/deploy/tron2_deploy.server.example_CN.yaml`,
  `configs/deploy/tron2_deploy.client.example_CN.yaml`

Future public tasks should follow the same naming pattern:
`configs/deploy/<task>_server.yaml` and `configs/deploy/<task>_client.yaml`.

For custom private profiles, copy the generic templates to `.local.yaml` files
and edit only those local files:

```bash
cp configs/deploy/tron2_deploy.server.example.yaml configs/deploy/my_task_server.local.yaml
cp configs/deploy/tron2_deploy.client.example.yaml configs/deploy/my_task_client.local.yaml
```

Do not commit `.local.yaml` files. They are for private paths, robot addresses,
Bridge hosts, camera serial numbers, and local-only experiments.

Minimal server profile:

```yaml
policy:
  config: TASK_CONFIG_NAME
  repo_id: DATASET_REPO_ID
  checkpoint_dir: /path/to/checkpoints/TASK_CONFIG_NAME/experiment/step
  default_prompt: TASK_PROMPT

server:
  host: 0.0.0.0
  port: 8000
```

Minimal client profile:

```yaml
client:
  task: TASK_NAME
  policy_host: 127.0.0.1
  policy_port: 8000
  observation_source: bridge
  rtc_enabled: true

robot:
  ip: ROBOT_IP
```

Server fields:

| Field | Description |
| --- | --- |
| `policy.config` | Training config name registered in `src/openpi/training/config.py`. |
| `policy.repo_id` | Asset directory name used to load normalization statistics. |
| `policy.checkpoint_dir` | Path to the trained checkpoint step directory. |
| `policy.default_prompt` | Default language instruction when the client does not pass `--prompt`. |
| `policy.record` | Saves raw policy inputs/outputs for debugging when `true`. |
| `policy.action_horizon` | Optional inference action chunk length override. |
| `policy.state_dim` | Optional TRON2 state/action output dimension override. |
| `policy.use_delta_joint_actions` | Optional override for delta-action transforms. |
| `server.host` / `server.port` | Policy server listen address. |

Client fields:

| Field | Description |
| --- | --- |
| `client.task` | Human-readable task name used for record filenames. |
| `client.policy_host` / `client.policy_port` | Policy server address from the client process. |
| `client.observation_source` | `bridge` or `legacy`. |
| `client.state_dim` | Physical state/action vector width; must match the module layout and training config. |
| `client.state_layout` | Component order in the flat vector; explicit module configs must match the derived layout. |
| `client.fps` | Policy action playback rate. |
| `client.publish_rate` | Background ServoJ command publication rate. |
| `client.max_steps` | Number of policy chunks to run in non-RTC mode; `null` means run until stopped. |
| `client.rtc_enabled` | Use `pi_client_rtc.py` when `true`; use `pi_client.py` when `false`. |
| `client.duration` | RTC runtime in seconds; `0` means run until stopped. |
| `client.execution_horizon` / `client.delay` | RTC `s` and initial `d` timing values. |
| `client.rtc_guidance_enabled` | Enables inference-time RTC VJP guidance. |
| `client.trained_rtc_mode` | Uses training-time RTC conditioning when the checkpoint was trained for it. |
| `robot.ip` / `robot.port` | TRON2 WebSocket controller address. |
| `bridge.host` | TRON2 Bridge WebSocket host when using bridge observations. |
| `camera.serial_to_name` | RealSense serial-to-policy-camera-name mapping when using legacy mode. |

#### Modular Hardware Layouts

The generic client template contains examples for the `arm`, `end_effector`,
`head`, and `mobile_base` sections. Leaving all four sections unspecified keeps
the legacy-compatible ServoJ + paired-gripper layout. Supplying any section
enables explicit module selection and validates `client.state_dim` and
`client.state_layout` against the derived layout.

| Module combination | Physical dimension |
| --- | ---: |
| ServoJ/ServoP + paired grippers | 16 |
| ServoJ/ServoP + paired grippers + head | 18 |
| ServoJ/ServoP + paired grippers + chassis/lifter | 19 |
| ServoJ/ServoP + paired grippers + head + chassis/lifter | 21 |
| ServoJ/ServoP + paired BrainCo2 hands | 26 |
| ServoJ/ServoP + paired BrainCo2 hands + head | 28 |
| ServoJ/ServoP + paired BrainCo2 hands + chassis/lifter | 29 |
| ServoJ/ServoP + paired BrainCo2 hands + head + chassis/lifter | 31 |

ServoJ and ServoP both use seven values per arm, but their meanings differ:
ServoJ contains joint targets, while ServoP contains
`[x, y, z, qw, qx, qy, qz]` end-effector poses. ServoP quaternions must not be
linearly blended. Each BrainCo2 hand uses six values and requires
`end_effector.command_time`. The chassis/lifter group uses three values:
`[chassis_linear_x, chassis_angular_z, lifter_state]`.

When Bridge supplies images, ServoP or BrainCo2 requires
`bridge.state_source: legacy` so physical state is read from the robot
WebSocket. Lifter control requires
`mobile_base.lifter_state_source: position_mm`; `raw_q` is observation-only.
The public runtime supports the WebSocket control backend only. See
`configs/deploy/tron2_deploy.client.example.yaml` for all fields and comments.

`policy.repo_id` must match the asset directory inside the checkpoint:

```text
checkpoint_dir/assets/<policy.repo_id>/norm_stats.json
```

Example TRON2 config names currently registered in the code:

| `policy.config` | `policy.repo_id` |
| --- | --- |
| `pi05_tron2_alarm` | `alarm` |
| `pi05_tron2_banana` | `banana` |
| `pi05_tron2_cabinet` | `cabinet` |
| `pi05_tron2_candy` | `candy` |
| `pi05_tron2_chess` | `chess` |
| `pi05_tron2_cloth` | `cloth` |
| `pi05_tron2_drawer` | `drawer` |
| `pi05_tron2_duck` | `duck` |
| `pi05_tron2_sort` | `sort` |

### Run Policy Serving

Start the policy server:

```bash
uv run scripts/serve_policy.py \
  --profile configs/deploy/candy_server.yaml
```

**Note: the robot should be in the initial state after L1+X, then switched to
advanced developer mode. After the client starts, the robot will spread both
arms sideways and then move them forward. If the robot is not in the initial
state, it may lift both arms directly; keep the front workspace clear.**

For a non-RTC client profile with `client.rtc_enabled: false`, run:

```bash
uv run examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml
```

Override the prompt for one run:

```bash
uv run examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml \
  --prompt="put the object into the drawer"
```

Stop the client manually when `client.max_steps` is `null`.

### RTC Deployment

RTC uses the same server command. The server detects whether the loaded model
supports RTC and publishes `rtc_enabled` plus `action_horizon` in websocket
metadata. The client supplies runtime timing from the YAML.

Set these fields in your task client profile:

```yaml
client:
  rtc_enabled: true
  duration: 120
  fps: 30
  execution_horizon: 10
  delay: 2
  rtc_guidance_enabled: true
  rtc_guidance_weight: 10.0
  trained_rtc_mode: false
```

Then run:

```bash
uv run examples/tron2/pi_client_rtc.py \
  --profile configs/deploy/candy_client.yaml
```

The RTC client warms up the model, seeds the action queue, retries short
observation timeouts without reusing stale observations, records queue
diagnostics, and can apply optional client-side action smoothing through
`client.rtc_action_postprocess`.

## Training A New TRON2 Task

For public task configs, prefer the YAML entry point instead of editing
`src/openpi/training/config.py`:

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
```

Edit `configs/train/tron2_tasks/my_task.yaml`, then point LeRobot at your dataset
root. If `repo_id: my_dataset`, the dataset should normally be available under
`$HF_LEROBOT_HOME/my_dataset/` with `data/` and `meta/` subdirectories:

```bash
export HF_LEROBOT_HOME=/path/to/datasets
```

Set `fsdp_devices` in the task YAML to the number of devices used by each FSDP
shard. Keep it at `1` for single-device training; for multi-device training, the
value must divide the number of JAX devices visible to the process.

`state_dim` and `action_dim` must currently be equal. They must also match the
dataset state/action width, normalization statistics, checkpoint, the server
profile's `policy.state_dim`, and the client's physical layout. Choose modular
task dimensions from the table above.

Compute normalization statistics before the first training run:

```bash
uv run scripts/compute_norm_stats.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

Then start training:

```bash
uv run scripts/train_tron2_task.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

For one-command training, use `scripts/cloud_train_entrypoint_portable.sh`. It
computes normalization statistics first unless `--skip-norm` is passed, then
launches training.
Its `--physical-dim` option writes both `state_dim` and `action_dim` into the
generated task config. The default is `16`, and the valid range is 1 through 32.

Cloud/platform mode assumes the dataset and weights are mounted by the platform.
With the default paths, `--repo-id input` means the LeRobot dataset is under
`/data/input/`, the initial params are at `/data/checkpoint/params`, and outputs
go to `/data/output`:

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --repo-id input \
  --exp my_task \
  --prompt "perform the configured task" \
  --max-frames 100000
```

Local/custom-path mode passes the dataset root and weight path explicitly:

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --data-dir /path/to/datasets \
  --repo-id my_dataset \
  --weight /path/to/checkpoint/params \
  --output-dir /path/to/output \
  --exp my_task \
  --prompt "perform the configured task" \
  --max-frames 100000
```

For a 19D chassis/lifter task, add `--physical-dim 19` to either generated-config
command above. Existing 16D tasks retain their default when the option is omitted.

You can also run the portable entrypoint with an edited task YAML. In that mode,
the YAML controls `repo_id`, `weight_loader`, `assets_base_dir`, and
`checkpoint_base_dir`; `--data-dir` still sets `HF_LEROBOT_HOME`:

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --task-config configs/train/tron2_tasks/my_task.yaml \
  --exp my_task \
  --data-dir "$HF_LEROBOT_HOME" \
  --max-frames 100000
```

Real task YAML files are ignored by `.gitignore`; keep only
`configs/train/tron2_tasks/example.yaml` in the public repository. The template
supports `repo_id`, prompt, dataset column keys, `action_horizon`, `state_dim`,
`action_dim`, `fsdp_devices`, base checkpoint weights, output directories,
`prompt_from_task`, and optional `rtc_training_simulated_delay`.

## Network Deployment Boundary

Run the policy server, TRON2 robot control, and Bridge observation paths only on
a controlled robot LAN that is accessible to authorized systems. Do not expose
these interfaces to the Internet or to an untrusted shared network.

Some runtime links may not provide application authentication or TLS. A `wss://`
Bridge endpoint does not secure the other links. Internet-facing, cross-site, or
cloud robot-control topologies require a separate security review before use.
Source disclosure is not a functional safety approval or real-robot certification.
See `SECURITY.md` for vulnerability reporting and the full deployment boundary.

## Safety Notes

- Run real-robot clients only with a trained operator present and emergency stop
  access available.
- Verify `robot.init_joints`, `robot.init_head`, endpoint addresses, and camera
  ordering before executing policy actions.
- Keep the workspace clear and start with short `client.max_steps` values before
  longer runs.
- This repository does not include private low-level safety controllers or a
  real-robot safety certification.

## Troubleshooting

- If `uv sync` or `uv run` is slow, check package index and network access.
- If the policy cannot load, verify `policy.config`, `policy.repo_id`, and
  `policy.checkpoint_dir`.
- If normalization stats are missing, check
  `checkpoint_dir/assets/<policy.repo_id>/norm_stats.json`.
- If the client cannot connect to the policy server, verify
  `client.policy_host` and `client.policy_port`.
- If bridge observations time out, verify `bridge.host`, TLS settings, and
  Bridge availability.
- If legacy mode misses cameras, verify RealSense serial numbers in
  `camera.serial_to_name`.
- If the robot does not move, verify `robot.ip`, `robot.port`, controller state,
  and that the workspace is safe before retrying.

## Third-Party Origins

This repository is derived from OpenPI and retains upstream OpenPI components.
Some files also include code adapted from Big Vision, HuggingFace Transformers,
LeRobot RTC, Physical Intelligence Kinetix, and `msgpack-numpy`. OpenPI commit
`e01d2290dfef823304b9a59a94b29e5945e38b2d` was used as the working baseline;
the exact origin of each component is not claimed. See `NOTICE`,
`THIRD_PARTY_NOTICES.md`, and `MODIFICATIONS.md` for paths, sources, licenses,
and modification status.

## Contributing

Contributions should stay within the public deployment scope above. Do not
submit private robot profiles, real credentials, internal URLs, customer data,
datasets, model weights, or logs. See `CONTRIBUTING.md` for the base
contribution guidelines.

## License

Source code follows its file-level source licenses. Unless otherwise noted,
project source is provided under the Apache License 2.0 in `LICENSE`;
third-party source remains under the licenses recorded in
`THIRD_PARTY_NOTICES.md` and `LICENSES/`.

`LICENSE_GEMMA.txt` is retained byte-for-byte as upstream-carried model asset
terms material. This source snapshot includes no Gemma or PaliGemma weights,
checkpoints, or model derivatives. External model assets and derivatives use
the applicable Gemma Terms. Those terms do not relicense or add restrictions
to Apache source code. Publishing a model asset, model derivative, or Hosted
Service requires re-review.
