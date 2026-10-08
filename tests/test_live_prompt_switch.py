"""Offline checks for switching prompts while the policy client is running."""

from pathlib import Path
import sys
from threading import Event

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "tron2"))

from deploy_config import PromptController
from pi_client_rtc import ActionPostprocessConfig
from pi_client_rtc import ActionPostProcessor
from pi_client_rtc import inference_producer
from tron2_env.rtc import ActionQueue
from tron2_env.rtc import LatencyTracker


def test_prompt_change_clears_queue_and_rejects_stale_result():
    prompt = PromptController("old subtask")
    queue = ["old action"]
    prompt.set_on_change(queue.clear)
    _, old_version = prompt.snapshot()

    assert not prompt.set(" old subtask ")
    assert queue == ["old action"]
    assert prompt.set("new subtask")
    assert queue == []

    accepted, _ = prompt.apply_if_current(old_version, lambda: queue.append("late old result"))
    assert not accepted
    assert queue == []

    _, new_version = prompt.snapshot()
    accepted, _ = prompt.apply_if_current(new_version, lambda: queue.append("new result"))
    assert accepted
    assert queue == ["new result"]


def test_rtc_producer_retries_with_new_prompt_after_inflight_switch():
    prompt = PromptController("old subtask")
    queue = ActionQueue(rtc_enabled=True)
    prompt.set_on_change(queue.clear)
    stop = Event()
    requested_prompts = []

    class FakeEnv:
        def get_obs(self):
            return {
                "state": np.zeros(18, dtype=np.float32),
                "images": {name: np.zeros((16, 16, 3), dtype=np.uint8) for name in (
                    "cam_high", "cam_left_wrist", "cam_right_wrist"
                )},
            }

    class FakePolicy:
        def infer(self, obs, **kwargs):
            requested_prompts.append(obs["prompt"])
            if len(requested_prompts) == 1:
                prompt.set("new subtask")
            else:
                stop.set()
            value = float(len(requested_prompts))
            return {
                "raw_actions": np.full((30, 32), value, dtype=np.float32),
                "actions": [np.full(18, value, dtype=np.float32) for _ in range(30)],
            }

    inference_producer(
        FakePolicy(), FakeEnv(), queue, LatencyTracker(), stop,
        fps=30.0, execution_horizon=10, delay=6,
        rtc_guidance_enabled=False, rtc_guidance_weight=10.0,
        trigger_queue_size=20,
        action_postprocessor=ActionPostProcessor(ActionPostprocessConfig()),
        action_horizon=30, trained_rtc_mode=True, prompt_controller=prompt,
    )

    assert requested_prompts == ["old subtask", "new subtask"]
    assert queue.qsize() > 0
    assert np.all(queue.get() == 2.0)


def test_rtc_consumer_does_not_execute_action_dequeued_during_switch():
    from pi_client_rtc import control_consumer

    prompt = PromptController("old subtask")
    stop = Event()
    executed = []

    class FakeQueue:
        def qsize(self):
            return 1

        def get_action_index(self):
            return 0

        def get(self):
            prompt.set("new subtask")
            stop.set()
            return np.ones(18, dtype=np.float32)

    class FakeEnv:
        def step(self, action):
            executed.append(action)

    control_consumer(FakeEnv(), FakeQueue(), stop, fps=30.0, prompt_controller=prompt)
    assert executed == []
