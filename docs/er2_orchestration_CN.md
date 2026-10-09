# ER 2 任务编排

本仓库可以选择让 Gemini Robotics ER 2 作为 TRON2 π0.5 VLA 上层的任务编排器。
只有部署 profile 中明确设置 `er2.enabled: true` 时才会启用，默认客户端行为不变。

ER2 只能从 `er2.skills` 的封闭技能目录中选择 ID。被选中的固定 `prompt` 会交给现有
RTC 客户端。prompt 变化时，现有的 prompt 版本机制会清空动作队列，并丢弃旧 prompt
产生的进行中结果。ER2 不接收也不输出机器人电机指令；策略客户端负责推理，机器人环境
负责执行。

安装可选 SDK，并把 API key 放在仓库外：

```bash
uv sync --extra er2
export GEMINI_API_KEY='...'
```

在受控机器人局域网中启动：

```bash
uv run python examples/tron2/pi_client_rtc.py \
  --profile configs/deploy/ruyi_er2_client.example.yaml
```

profile 中的 `er2.task` 是高层任务。每个技能需要 ID、训练时使用的 VLA prompt，以及
可观察的完成条件。ER2 可以返回 `keep`、`set_skill`、`uncertain`、`failure` 或 `done`；
只有目录中的 `set_skill` 会更新 VLA prompt。监视线程最多每个 `interval_s` 发送一次最新
`cam_high` 画面，并使用只保留最新观测的缓冲区，避免旧画面堆积。

示例 profile 只展示接线方式，不构成功能安全认证。上真机前仍需验证相机名称、机器人地址、
checkpoint、技能能力、API 延迟、完成判据和停止行为。不要提交 API key、私有机器人地址、
checkpoint 路径或 `.local.yaml` 配置。
