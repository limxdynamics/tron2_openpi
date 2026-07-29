# TRON2 OpenPI 仓库

[English](README.md) | [安装指南](INSTALL_CN.md)

`tron2_openpi` 是基于 OpenPI 改进的 TRON2 部署仓库。它保留 OpenPI 的
policy serving、pi0/pi0.5 模型栈和客户端基础能力，并加入 TRON2 policy
transform、部署配置模板和 TRON2 真机客户端示例。

本仓库需要和同级的 `tron2_env` 运行时包一起使用。它的定位是集成和部署示例，
不是私有 checkpoint、数据集、low-level 机器人 SDK 或本地部署配置的完整发布。

## 本仓库包含什么

- 通过 `scripts/serve_policy.py` 启动 pi0.5 策略服务。
- `src/openpi/policies/tron2_policy.py` 中的 TRON2 policy 输入/输出转换。
- `src/openpi/training/config.py` 中的 TRON2 训练/部署配置注册。
- `examples/tron2/` 中的 TRON2 真机客户端。
- `configs/deploy/` 中的公开部署配置模板。
- 通过 `scripts/train_tron2_task.py` 和 `configs/train/tron2_tasks/example.yaml`
  使用 YAML 配置新的 TRON2 训练任务。
- 可选的 Bridge 观测模式：从 TRON2 Bridge 获取图像和状态。
- 可选的 legacy RealSense 观测模式：使用本机直连相机。
- RTC 部署客户端，包含 warmup、观测超时恢复、队列诊断和可选动作平滑。
- `packages/openpi-client/` 中的 OpenPI client 包。

## 本仓库不包含什么

- 模型权重和 checkpoint 目录。
- 训练数据集、评测数据集、日志或 benchmark 结果。
- 私有 `.local.yaml` 部署文件。
- 凭据、真实相机序列号、客户数据或私有本地部署配置。
- 尚未开发完成的 low-level 机器人 transport。
- 无人值守真机运行的安全认证。

## 示例任务与公开资源

以下公开任务资源会随更多示例发布持续更新。

| 任务 | 用户文档 | 模型权重 | 公开部署 profile |
| --- | --- | --- | --- |
| Candy | [TRON2 OpenPI Candy 用户文档](https://cwjgfm21di.feishu.cn/wiki/DitfwjCRiiSWhBk3MTUcA14tnsh) | [limx-tron2-dev/tron2-openpi-models](https://huggingface.co/limx-tron2-dev/tron2-openpi-models) | `configs/deploy/candy_server.yaml`, `configs/deploy/candy_client.yaml` |

模型权重和 checkpoint 不存放在本仓库中。使用真机任务前，请先从表格中的模型仓库下载
权重，并在对应任务 server profile 中填写实际 checkpoint 路径。

## 目录结构

```text
同级目录/
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

请保持 `tron2_openpi/` 和 `tron2_env/` 两个目录同级。TRON2 客户端启动时会把
同级 `../tron2_env/src` 加入 `sys.path`，因此可以直接导入运行时包。
录制动作回放工具位于 `../tron2_env/examples/replay_data.py`。

## 安装

环境要求和安装命令已移到 [INSTALL_CN.md](INSTALL_CN.md)。英文版见
[INSTALL.md](INSTALL.md)。

## 部署

### 部署配置

每个任务使用两份 profile：

- server profile：模型 checkpoint、默认 prompt、policy 覆盖项和服务端口。
- client profile：policy server 地址、机器人地址、观测来源、相机/Bridge、执行循环和 RTC 参数。

请从公开 Candy profile 或通用模板开始：

- Candy server profile：`configs/deploy/candy_server.yaml`
- Candy client profile：`configs/deploy/candy_client.yaml`
- 英文模板：`configs/deploy/tron2_deploy.server.example.yaml`、
  `configs/deploy/tron2_deploy.client.example.yaml`
- 中文模板：`configs/deploy/tron2_deploy.server.example_CN.yaml`、
  `configs/deploy/tron2_deploy.client.example_CN.yaml`

后续公开任务按同样方式命名：
`configs/deploy/<任务名>_server.yaml` 和 `configs/deploy/<任务名>_client.yaml`。

如果需要自定义私有配置，可以从通用模板复制 `.local.yaml`，并只修改本地文件：

```bash
cp configs/deploy/tron2_deploy.server.example_CN.yaml configs/deploy/my_task_server.local.yaml
cp configs/deploy/tron2_deploy.client.example_CN.yaml configs/deploy/my_task_client.local.yaml
```

不要提交 `.local.yaml` 文件。这类文件用于保存私有路径、机器人地址、Bridge 地址、
相机序列号和本地实验配置。

最小 server profile：

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

最小 client profile：

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

server 字段：

| 字段 | 说明 |
| --- | --- |
| `policy.config` | `src/openpi/training/config.py` 中注册的训练配置名。 |
| `policy.repo_id` | 用于加载归一化统计的 assets 目录名。 |
| `policy.checkpoint_dir` | 训练好的 checkpoint step 目录。 |
| `policy.default_prompt` | 客户端未传 `--prompt` 时使用的默认语言指令。 |
| `policy.record` | 为 `true` 时保存原始 policy 输入/输出，用于调试。 |
| `policy.action_horizon` | 可选的推理 action chunk 长度覆盖项。 |
| `policy.state_dim` | 可选的 TRON2 state/action 输出维度覆盖项。 |
| `policy.use_delta_joint_actions` | 可选的 delta action transform 覆盖项。 |
| `server.host` / `server.port` | policy server 监听地址。 |

client 字段：

| 字段 | 说明 |
| --- | --- |
| `client.task` | 任务名，用于生成录制文件名。 |
| `client.policy_host` / `client.policy_port` | 客户端看到的 policy server 地址。 |
| `client.observation_source` | `bridge` 或 `legacy`。 |
| `client.state_dim` | `16` 表示双臂和夹爪，`18` 表示额外包含头部关节。 |
| `client.fps` | policy action 播放频率。 |
| `client.publish_rate` | 后台 ServoJ 指令发送频率。 |
| `client.max_steps` | 非 RTC 模式运行多少个 policy chunk；`null` 表示持续运行直到手动停止。 |
| `client.rtc_enabled` | 为 `true` 时使用 `pi_client_rtc.py`；为 `false` 时使用 `pi_client.py`。 |
| `client.duration` | RTC 运行时长，单位秒；`0` 表示一直运行。 |
| `client.execution_horizon` / `client.delay` | RTC 的 `s` 和初始 `d` 时序参数。 |
| `client.rtc_guidance_enabled` | 是否启用推理时 RTC VJP guidance。 |
| `client.trained_rtc_mode` | checkpoint 使用训练时 RTC 时打开该模式。 |
| `robot.ip` / `robot.port` | TRON2 WebSocket 机器人控制器地址。 |
| `bridge.host` | Bridge 观测模式使用的 TRON2 Bridge WebSocket 地址。 |
| `camera.serial_to_name` | legacy 模式下 RealSense 序列号到 policy 相机名的映射。 |

`policy.repo_id` 必须和 checkpoint 内的 assets 目录一致：

```text
checkpoint_dir/assets/<policy.repo_id>/norm_stats.json
```

当前代码中注册的 TRON2 示例配置：

| `policy.config` | `policy.repo_id` |
| --- | --- |
| `pi05_tron2_alarm` | `alarm` |
| `pi05_tron2_Banana` | `banana` |
| `pi05_tron2_cabinet` | `cabinet` |
| `pi05_tron2_Candy` | `candy` |
| `pi05_tron2_Chess` | `chess` |
| `pi05_tron2_Cloth` | `cloth` |
| `pi05_tron2_Drawer` | `drawer` |
| `pi05_tron2_Duck` | `duck` |
| `pi05_tron2_SortFruit` | `sort` |

### 启动策略服务

启动 policy server：

```bash
uv run --no-sync scripts/serve_policy.py \
  --profile configs/deploy/candy_server.yaml
```

**注意：机器人需处于L1+X后的初始状态，然后切换到高级开发者模式，运行客户端后机器人会侧展双臂，然后前伸，若非初始状态可能会直接抬起双臂，警惕前方物体风险！！！**

如果 client profile 中 `client.rtc_enabled: false`，在另一个终端运行普通客户端：

```bash
uv run --no-sync python examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml
```

临时覆盖任务指令：

```bash
uv run --no-sync python examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml \
  --prompt="put the object into the drawer"
```

当 `client.max_steps` 为 `null` 时，需要手动停止客户端。

### RTC 部署

RTC 使用同一个 server 命令。server 会自动检测加载的模型是否支持 RTC，并在
websocket metadata 中发布 `rtc_enabled` 和 `action_horizon`。client 侧运行参数来自
YAML。

在任务 client profile 中设置：

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

然后运行：

```bash
uv run --no-sync python examples/tron2/pi_client_rtc.py \
  --profile configs/deploy/candy_client.yaml
```

RTC client 会先 warmup 模型并填充 action queue；运行中会在短暂观测超时时等待新鲜
观测，不复用旧 obs；同时记录队列合并诊断，并可通过
`client.rtc_action_postprocess` 开启可选的动作平滑。

## 训练新的 TRON2 任务

公开任务配置建议使用 YAML 入口，而不是直接改 `src/openpi/training/config.py`：

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
```

修改 `configs/train/tron2_tasks/my_task.yaml` 后，先把 LeRobot 数据集根目录指向你
自己的数据集目录。如果 `repo_id: my_dataset`，通常需要存在
`$HF_LEROBOT_HOME/my_dataset/data/` 和 `$HF_LEROBOT_HOME/my_dataset/meta/`：

```bash
export HF_LEROBOT_HOME=/path/to/datasets
export HF_HUB_OFFLINE=1
```

首次训练前先计算 normalization statistics：

```bash
uv run --no-sync scripts/compute_norm_stats.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

然后启动训练：

```bash
uv run --no-sync scripts/train_tron2_task.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

如需一条命令完成训练，可以使用 `scripts/cloud_train_entrypoint_portable.sh`。
除非传入 `--skip-norm`，它会先计算 norm，再启动训练。

云端/平台挂载模式假设数据和权重由平台挂载。按默认路径，`--repo-id input` 表示
LeRobot 数据集位于 `/data/input/`，初始权重位于 `/data/checkpoint/params`，输出写到
`/data/output`：

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --repo-id input \
  --exp my_task \
  --prompt "perform the configured task" \
  --max-frames 100000
```

本地/自定义路径模式显式传入数据集根目录和权重路径：

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

也可以让一站式入口使用已经编辑好的任务 YAML。此模式下，YAML 控制 `repo_id`、
`weight_loader`、`assets_base_dir` 和 `checkpoint_base_dir`；`--data-dir` 仍用于设置
`HF_LEROBOT_HOME`：

```bash
scripts/cloud_train_entrypoint_portable.sh \
  --task-config configs/train/tron2_tasks/my_task.yaml \
  --exp my_task \
  --data-dir "$HF_LEROBOT_HOME" \
  --max-frames 100000
```

真实任务 YAML 已被 `.gitignore` 忽略；公开仓库只保留
`configs/train/tron2_tasks/example.yaml`。模板支持 `repo_id`、prompt、数据列名、
`action_horizon`、`state_dim`、base checkpoint 权重、输出路径，以及可选的
`prompt_from_task` 和 `rtc_training_simulated_delay`。

## 网络部署边界

policy server/client、TRON2 机器人控制和 Bridge 观测链路仅支持在受控机器人局域网
中使用，且只能由授权系统接入。不要把这些接口暴露到互联网或不受信任的共享网络。

部分运行时链路不一定提供应用层鉴权或 TLS。`wss://` Bridge 端点不会保护其他链路。
任何面向互联网、跨站点或云端的机器人控制拓扑，都需要在使用前单独进行安全评审。
源码公开不等于功能安全批准或真机认证。
漏洞报告方式和完整部署边界见 `SECURITY.md`。

## 安全注意事项

- 真机客户端只能在受过训练的操作人员在场、且急停可用时运行。
- 执行 policy 前确认 `robot.init_joints`、`robot.init_head`、端点地址和相机顺序正确。
- 保持机器人工作空间清空；首次运行先设置较小的 `client.max_steps`，确认行为后再延长运行。
- 本仓库不包含私有 low-level 安全控制器，也不提供真机安全认证。

## 常见问题

- 如果 `uv sync` 或 `uv run` 很慢，检查 Python 包索引和网络访问。
- 如果 policy 加载失败，检查 `policy.config`、`policy.repo_id` 和 `policy.checkpoint_dir`。
- 如果缺少归一化统计，检查 `checkpoint_dir/assets/<policy.repo_id>/norm_stats.json`。
- 如果客户端连不上 policy server，检查 `client.policy_host` 和 `client.policy_port`。
- 如果 Bridge 观测超时，检查 `bridge.host`、TLS 设置和 Bridge 服务状态。
- 如果 legacy 模式找不到相机，检查 `camera.serial_to_name` 中的 RealSense 序列号。
- 如果机器人不运动，检查 `robot.ip`、`robot.port`、控制器状态，并确认工作空间安全后再重试。

## 第三方来源

本仓库基于 OpenPI 派生，并保留上游 OpenPI 组件。部分文件还包含来自 Big Vision、
HuggingFace Transformers、LeRobot RTC、Physical Intelligence Kinetix 和
`msgpack-numpy` 的改编代码。OpenPI commit
`e01d2290dfef823304b9a59a94b29e5945e38b2d` 是本仓库使用的基线提交，
不表示每个组件的精确来源都已确认。路径、来源、许可证和修改状态见 `NOTICE`、
`THIRD_PARTY_NOTICES.md` 和 `MODIFICATIONS.md`。

## 贡献

贡献内容应保持在上述公开部署范围内。请不要提交私有机器人配置、真实凭据、内部
URL、客户数据、数据集、模型权重或日志。基础贡献说明见 `CONTRIBUTING.md`。

## 许可证

源码文件遵循各自的文件级源码许可证。除非另有说明，项目源码使用 `LICENSE` 中的
Apache License 2.0；第三方源码继续使用 `THIRD_PARTY_NOTICES.md` 和 `LICENSES/`
中记录的许可证。

`LICENSE_GEMMA.txt` 按上游原文逐字节保留，作为上游模型资产条款材料。当前源码
快照不包含 Gemma 或 PaliGemma 权重、checkpoint 或模型衍生物。外部模型资产及其
衍生物适用的 Gemma Terms 需单独遵守；这些条款不会重新许可 Apache 源码，也不会
给 Apache 源码增加限制。未来发布模型资产、模型衍生物或 Hosted Service 必须重新评审。
