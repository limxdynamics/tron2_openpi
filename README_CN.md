<div align="center">

<img src="docs/tron2_openpi_banner.png" alt="TRON2 OpenPI：面向真机操作的 VLA 策略" width="100%">

[English](README.md) | **中文** | [安装指南](INSTALL_CN.md)

[![Python](https://img.shields.io/badge/python-3.11-3776AB)](.python-version)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Models on Hugging Face](https://img.shields.io/badge/Hugging%20Face-models-FFD21E)](https://huggingface.co/limxdynamics/tron2-openpi-models)
[![Models on ModelScope](https://img.shields.io/badge/ModelScope-models-624AFF)](https://modelscope.cn/models/limxdynamics/tron2-openpi-models)
[![Runtime: tron2_env](https://img.shields.io/badge/runtime-tron2__env-0A7BBB)](https://github.com/limxdynamics/tron2_env)

</div>

# TRON2 OpenPI

`tron2_openpi` 是基于 OpenPI 改进的 TRON2 部署仓库。它保留 OpenPI 的
policy serving、pi0/pi0.5 模型栈和客户端基础能力，并加入 TRON2 policy
transform、部署配置模板和 TRON2 真机客户端示例。

本仓库需要和同级的 `tron2_env` 运行时包一起使用。它的定位是集成和部署示例，
不是私有 checkpoint、数据集、low-level 机器人 SDK 或本地部署配置的完整发布。

## 亮点

- **pi0 / pi0.5 跑在真机上。** 沿用 OpenPI 的策略服务和模型栈，通过 policy transform 和部署配置接入 TRON2。
- **6 个带权重的公开任务。** 抓糖果、叠衣服、分拣、桌面整理、倒豆子、清理地面，每个任务都有用户
  文档、公开权重和部署 profile，链接见下一节。
- **一个 YAML 文件微调新任务。** `scripts/train_tron2_task.py` 配合 `configs/train/tron2_tasks/example.yaml`。
- **平滑的实时执行。** RTC 客户端支持预热、观测超时恢复、队列诊断和可选的动作平滑。
- **模块化硬件。** 支持 ServoJ / ServoP 双臂、夹爪或 BrainCo2 灵巧手、头部、底盘和升降台。

## 示例任务与公开资源

以下公开任务资源会随更多示例发布持续更新。

| 任务 | 用户文档 | 模型权重 | 公开部署 profile |
| --- | --- | --- | --- |
| Candy | [TRON2 OpenPI Candy 用户文档](https://cwjgfm21di.feishu.cn/wiki/DitfwjCRiiSWhBk3MTUcA14tnsh) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/candy_server.yaml`, `configs/deploy/candy_client.yaml` |
| Cloth | [TRON2 OpenPI Cloth 用户文档](https://cwjgfm21di.feishu.cn/wiki/Bcw8wthgpiLrVWkHXk0cBfLOnnc) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/cloth_server.yaml`, `configs/deploy/cloth_client.yaml` |
| Sort | [TRON2 OpenPI Sort 用户文档](https://cwjgfm21di.feishu.cn/wiki/PTpAwY597isKvXkJFq3csvLbnMh) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/sort_server.yaml`, `configs/deploy/sort_client.yaml` |
| 桌面整理 | [TRON2 OpenPI 桌面整理用户文档](https://cwjgfm21di.feishu.cn/wiki/WWm9woWZri3wDYkL6NxcBeC2n4b) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/desktop_server.yaml`, `configs/deploy/desktop_client.yaml` |
| 倒豆子 | [TRON2 OpenPI 倒豆子用户文档](https://cwjgfm21di.feishu.cn/wiki/Fvycw6Xk5ieqnykYxFYcOouUnGg) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/brainco_server.yaml`, `configs/deploy/brainco_client.yaml` |
| 清理地面 | [TRON2 OpenPI 清理地面用户文档](https://cwjgfm21di.feishu.cn/wiki/YjhjwBunVi87S7kbJffcMuW5nSg) | [Hugging Face](https://huggingface.co/limxdynamics/tron2-openpi-models) / [ModelScope](https://modelscope.cn/models/limxdynamics/tron2-openpi-models) | `configs/deploy/chassis_server.yaml`, `configs/deploy/chassis_client.yaml` |

模型权重和 checkpoint 不存放在本仓库中。使用真机任务前，请先从表格中的模型仓库下载
权重，并在对应任务 server profile 中填写实际 checkpoint 路径。模型仓库里还有一套
`banana` 权重，尚未提供用户文档或公开 profile。

用户文档是飞书 wiki 页面：需要有可访问该知识库的 LimX 飞书账号；中英文各指向自己那份文档。

## 整体架构

<p align="center">
  <img src="docs/images/architecture_cn.png" alt="整体架构：任务微调、策略服务、搭载 tron2_env 的机器人客户端、TRON2 控制器与 Bridge，均位于受控机器人局域网内" width="100%">
</p>

机器人客户端通过同级的 [`tron2_env`](https://github.com/limxdynamics/tron2_env) 运行时完成运动控制和观测采集。

## 各部分在哪里

| 内容 | 位置 |
| --- | --- |
| pi0.5 策略服务 | `scripts/serve_policy.py`，由一份 profile YAML 驱动 |
| TRON2 policy 输入/输出转换 | `src/openpi/policies/tron2_policy.py` |
| TRON2 训练/部署配置注册 | `src/openpi/training/config.py` |
| 真机客户端（非 RTC / RTC） | `examples/tron2/pi_client.py`、`examples/tron2/pi_client_rtc.py` |
| 各任务 profile 与通用模板 | `configs/deploy/` |
| YAML 驱动的任务训练 | `scripts/train_tron2_task.py` 配合 `configs/train/tron2_tasks/example.yaml` |
| 观测来源 | `client.observation_source: bridge` 从 TRON2 Bridge 取图像与状态；`legacy` 读客户端本机相机 |
| 模块化硬件选择 | client profile 的 `arm`、`end_effector`、`head`、`mobile_base` 四段 |
| OpenPI client 包 | `packages/openpi-client/` |

## 本仓库不包含什么

- 模型权重和 checkpoint 目录。
- 训练数据集、评测数据集、日志或 benchmark 结果。
- 私有 `.local.yaml` 部署文件。
- 凭据、真实相机序列号、客户数据或私有本地部署配置。
- 尚未开发完成的 low-level 机器人 transport。
- 无人值守真机运行的安全认证。

## 目录结构

```text
同级目录/
├── tron2_openpi/
│   ├── configs/
│   │   ├── deploy/                      # <任务名>_server.yaml + <任务名>_client.yaml
│   │   │   ├── candy_*.yaml             #   抓糖果、叠衣服、分拣、桌面整理
│   │   │   ├── brainco_*.yaml           #   倒豆子
│   │   │   ├── chassis_*.yaml           #   清理地面
│   │   │   └── tron2_deploy.*.example*.yaml   # 通用模板，中英各一套
│   │   └── train/
│   │       └── tron2_tasks/
│   │           └── example.yaml
│   ├── examples/
│   │   ├── tron2/                       # TRON2 客户端
│   │   │   ├── deploy_config.py
│   │   │   ├── pi_client.py
│   │   │   └── pi_client_rtc.py
│   │   └── ...                          # 上游示例集，未改动
│   ├── packages/
│   │   └── openpi-client/
│   ├── scripts/
│   │   ├── cloud_train_entrypoint_portable.sh
│   │   ├── compute_norm_stats.py
│   │   ├── docker/                      # 服务镜像与 compose 文件
│   │   ├── serve_policy.py
│   │   ├── train.py                     # 上游训练入口
│   │   ├── train_pytorch.py
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
[INSTALL.md](INSTALL.md)。如需在容器里跑 policy server，可构建
`scripts/docker/serve_policy.Dockerfile`，或执行
`docker compose -f scripts/docker/compose.yml up --build`。

## 其他文档

| 文档 | 内容 | 说明 |
| --- | --- | --- |
| [`docs/norm_stats.md`](docs/norm_stats.md) | 状态与动作归一化的原理，以及预训练统计量的来源 | 上游原文；其中列出的机器人和 assets id 属于上游，不是 TRON2 |
| [`docs/remote_inference.md`](docs/remote_inference.md) | 在一台机器上做服务、在另一台机器里嵌入客户端 | 上游原文；其中的 `--env` 是上游环境的用法，本 fork 用 `--profile` |
| [`docs/docker.md`](docs/docker.md) | 上游示例集的 Docker 说明 | 针对上游示例；`examples/tron2/` 下没有 `compose.yml` |
| [`SECURITY.md`](SECURITY.md) | 漏洞报告方式与完整部署边界 | — |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 哪些内容属于公开贡献范围 | — |

## 部署

### 部署配置

每个任务使用两份 profile：

- server profile：模型 checkpoint、默认 prompt、policy 覆盖项和服务端口。
- client profile：policy server 地址、机器人地址、观测来源、相机/Bridge、执行循环和 RTC 参数。

每个公开任务都自带一对 profile，不必先写 profile 就能启动：

| 任务 | server profile | client profile |
| --- | --- | --- |
| 抓糖果 | `configs/deploy/candy_server.yaml` | `configs/deploy/candy_client.yaml` |
| 叠衣服 | `configs/deploy/cloth_server.yaml` | `configs/deploy/cloth_client.yaml` |
| 分拣 | `configs/deploy/sort_server.yaml` | `configs/deploy/sort_client.yaml` |
| 桌面整理 | `configs/deploy/desktop_server.yaml` | `configs/deploy/desktop_client.yaml` |
| 倒豆子 | `configs/deploy/brainco_server.yaml` | `configs/deploy/brainco_client.yaml` |
| 清理地面 | `configs/deploy/chassis_server.yaml` | `configs/deploy/chassis_client.yaml` |

没有公开 profile 的任务，从通用模板开始：

- 英文模板：`configs/deploy/tron2_deploy.server.example.yaml`、
  `configs/deploy/tron2_deploy.client.example.yaml`
- 中文模板：`configs/deploy/tron2_deploy.server.example_CN.yaml`、
  `configs/deploy/tron2_deploy.client.example_CN.yaml`

新公开任务按 `<任务名>_server.yaml` / `<任务名>_client.yaml` 命名；`brainco_*` 与
`chassis_*` 沿用发布时的硬件命名。

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
| `policy.repo_id` | checkpoint 内保存归一化统计的 assets 目录名，即 `<policy.checkpoint_dir>/assets/<policy.repo_id>/norm_stats.json`。已发布的 checkpoint 用的是 `input`，但 chassis 那套是 `chassis270` —— 这一项必须和你实际下载的 checkpoint 里的目录名一致。 |
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
| `client.state_dim` | 物理 state/action 向量维度；必须与模块布局和训练配置一致。 |
| `client.state_layout` | 平铺向量的组件顺序；显式模块配置时必须与模块推导结果一致。 |
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

#### 模块化硬件布局

<p align="center">
  <img src="docs/images/state_layout_cn.png" alt="16、19、28、31 维模块组合对应的状态与动作向量布局" width="100%">
</p>

通用 client 模板包含 `arm`、`end_effector`、`head` 和 `mobile_base` 配置示例。
四个区块均不配置时，继续使用兼容旧任务的 ServoJ + 双夹爪布局；配置任一区块后，
客户端会按显式模块推导布局并校验 `client.state_dim` 和 `client.state_layout`。

| 模块组合 | 物理维度 |
| --- | ---: |
| ServoJ/ServoP + 双夹爪 | 16 |
| ServoJ/ServoP + 双夹爪 + 头部 | 18 |
| ServoJ/ServoP + 双夹爪 + 底盘/升降台 | 19 |
| ServoJ/ServoP + 双夹爪 + 头部 + 底盘/升降台 | 21 |
| ServoJ/ServoP + BrainCo2 双灵巧手 | 26 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 头部 | 28 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 底盘/升降台 | 29 |
| ServoJ/ServoP + BrainCo2 双灵巧手 + 头部 + 底盘/升降台 | 31 |

ServoJ 和 ServoP 每条手臂都占 7 维，但语义不同：ServoJ 是关节目标，ServoP 是
`[x, y, z, qw, qx, qy, qz]` 末端位姿。ServoP 的四元数不得使用线性动作混合。
BrainCo2 每只手占 6 维，必须配置 `end_effector.command_time`。底盘/升降台占 3 维：
`[chassis_linear_x, chassis_angular_z, lifter_state]`。

使用 Bridge 图像时，ServoP 或 BrainCo2 必须设置 `bridge.state_source: legacy`，让物理
状态从机器人 WebSocket 读取。启用升降台控制时必须设置
`mobile_base.lifter_state_source: position_mm`；`raw_q` 仅可用于观测。公开运行时只支持
WebSocket 控制后端。完整字段和注释见
`configs/deploy/tron2_deploy.client.example_CN.yaml`。

#### 配置名与 assets 目录

`policy.repo_id` 必须和 checkpoint 内的 assets 目录一致：

```text
checkpoint_dir/assets/<policy.repo_id>/norm_stats.json
```

下表中是 `src/openpi/training/config.py` 里注册的配置名，以及各自在代码里默认的 assets 目录名。
profile 可以覆盖这两项：公开任务 profile 用的是 `policy.repo_id: input`，并带上与 checkpoint
匹配的 `state_dim`。

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
| `pi05_tron2_example` | `example_task` |

### 启动策略服务

```bash
uv run scripts/serve_policy.py \
  --profile configs/deploy/candy_server.yaml
```

### 运行真机客户端

**注意：机器人需处于L1+X后的初始状态，然后切换到高级开发者模式，运行客户端后机器人会侧展双臂，然后前伸，若非初始状态可能会直接抬起双臂，警惕前方物体风险！！！**

如果 client profile 中 `client.rtc_enabled: false`，在另一个终端运行普通客户端：

```bash
uv run examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml
```

临时覆盖任务指令：

```bash
uv run examples/tron2/pi_client.py \
  --profile configs/deploy/my_task_client.local.yaml \
  --prompt="put the object into the drawer"
```

当 `client.max_steps` 为 `null` 时，需要手动停止客户端。

### RTC 部署

<p align="center">
  <img src="docs/images/rtc_cn.png" alt="RTC 时序：队列降到 H − s 时触发推理，推理约 d 步，引导权重衰减，然后合并" width="100%">
</p>

<sub>RTC 客户端如何让推理与执行并行。图中数值取自 `configs/deploy/candy_client.yaml` 和 `candy_server.yaml`。</sub>

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
uv run examples/tron2/pi_client_rtc.py \
  --profile configs/deploy/candy_client.yaml
```

RTC client 会先 warmup 模型并填充 action queue；运行中会在短暂观测超时时等待新鲜
观测，不复用旧 obs；同时记录队列合并诊断，并可通过
`client.rtc_action_postprocess` 开启可选的动作平滑。

## 训练新的 TRON2 任务

<p align="center">
  <img src="docs/images/training_cn.png" alt="训练流程：任务 YAML、归一化统计、微调训练、部署" width="100%">
</p>

公开任务配置建议使用 YAML 入口，而不是直接改 `src/openpi/training/config.py`：

```bash
cp configs/train/tron2_tasks/example.yaml configs/train/tron2_tasks/my_task.yaml
```

修改 `configs/train/tron2_tasks/my_task.yaml` 后，先把 LeRobot 数据集根目录指向你
自己的数据集目录。如果 `repo_id: my_dataset`，通常需要存在
`$HF_LEROBOT_HOME/my_dataset/data/` 和 `$HF_LEROBOT_HOME/my_dataset/meta/`：

```bash
export HF_LEROBOT_HOME=/path/to/datasets
```

任务 YAML 中的 `fsdp_devices` 表示每个 FSDP shard 使用的设备数。单设备训练保持
为 `1`；多设备训练时，该值必须能够整除当前进程可见的 JAX 设备总数。

`state_dim` 和 `action_dim` 当前必须相等，且必须与数据集中的 state/action 宽度、
归一化统计、checkpoint、server profile 的 `policy.state_dim` 以及 client 的物理布局
保持一致。模块化任务可按上面的维度表选择数值。

首次训练前先计算 normalization statistics：

```bash
uv run scripts/compute_norm_stats.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

然后启动训练：

```bash
uv run scripts/train_tron2_task.py \
  --task-config configs/train/tron2_tasks/my_task.yaml
```

如需一条命令完成训练，可以使用 `scripts/cloud_train_entrypoint_portable.sh`。
除非传入 `--skip-norm`，它会先计算 norm，再启动训练；`--prompt` 为必填，除非传入
`--prompt-from-task`，此时改用每个 episode 自带的任务文本。该脚本的 `--physical-dim`
会同时设置生成任务配置中的 `state_dim` 和 `action_dim`，默认值为 `16`，有效范围为
1 到 32。

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

例如 19D 底盘/升降台任务，在上述生成配置的命令中追加
`--physical-dim 19`。16D 旧任务不传该参数即可保持原有默认值。

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
`configs/train/tron2_tasks/example.yaml`。模板支持 `name`（决定 checkpoint 子目录名）、
`repo_id`、prompt、数据列名、`action_horizon`、`state_dim`、`action_dim`、
`fsdp_devices`、base checkpoint 权重、输出路径，以及可选的 `prompt_from_task` 和
`rtc_training_simulated_delay`。

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
- 如果服务启动时报 `Config '<name>' not found`，说明该 `policy.config` 未注册；可用的名字见上表，
  以及 `src/openpi/training/config.py` 里的 `_tron2_checkpoint_task_configs()`。
- 如果归一化统计加载失败，确认 `policy.repo_id` 指向 `<policy.checkpoint_dir>/assets/` 下实际存在的
  目录；公开任务 checkpoint 里的是 `input`。

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
