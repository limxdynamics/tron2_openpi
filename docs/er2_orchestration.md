# ER 2 task orchestration

This repository can optionally run Gemini Robotics ER 2 as a high-level task
orchestrator above the TRON2 π0.5 VLA. The integration is disabled unless a
profile contains `er2.enabled: true`.

ER 2 selects one ID from the closed `er2.skills` catalog. The selected skill's
fixed `prompt` is passed to the existing RTC client. A prompt change increments
the existing prompt version, clears queued actions, and causes in-flight
results from the previous prompt to be discarded. ER 2 never receives or emits
robot motor commands. The policy client remains responsible for inference and
the robot environment remains responsible for execution.

Install the optional SDK and keep the API key outside the repository:

```bash
uv sync --extra er2
export GEMINI_API_KEY='...'
```

Start a configured RTC client on a controlled robot LAN:

```bash
uv run python examples/tron2/pi_client_rtc.py \
  --profile configs/deploy/ruyi_er2_client.example.yaml
```

The profile's `er2.task` is the high-level goal. Each skill needs an ID, the
exact VLA prompt used during training, and an observable completion condition.
ER 2 may return `keep`, `set_skill`, `uncertain`, `failure`, or `done`; only a
catalogued `set_skill` updates the VLA prompt. The monitor sends the latest
`cam_high` image at most once per `interval_s` and uses a latest-only buffer so
stale images cannot build up.

The example profile is a wiring example, not a real-robot safety certificate.
Validate camera names, robot addresses, checkpoints, skill capabilities, API
latency, completion criteria, and stop behavior before physical use. Do not
commit API keys, private robot addresses, checkpoint paths, or `.local.yaml`
profiles.
