"""Offline tests for the optional ER 2 orchestration boundary."""

import base64
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from openpi_client.er2 import ER2Orchestrator, ER2Skill, LatestObservationBuffer, encode_cam_high_jpeg


@dataclass
class _FakeResponse:
    id: str
    arguments: dict

    @property
    def steps(self):
        return [
            SimpleNamespace(
                type="function_call",
                name="update_execution",
                id=f"call-{self.id}",
                arguments=self.arguments,
            )
        ]


class _FakeInteractions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        args = {
            "decision": "set_skill" if len(self.calls) == 1 else "done",
            "skill_id": "open_drawer" if len(self.calls) == 1 else "none",
            "previous_skill_status": "not_started" if len(self.calls) == 1 else "completed",
            "completion_time_seconds": -1 if len(self.calls) == 1 else 2,
            "evidence": "The requested visible state is present.",
            "remaining_plan": [] if len(self.calls) > 1 else ["close_drawer"],
            "goal_status": "in_progress" if len(self.calls) == 1 else "achieved",
        }
        return _FakeResponse(f"interaction-{len(self.calls)}", args)


class _FakeClient:
    def __init__(self):
        self.interactions = _FakeInteractions()


def _orchestrator(client=None):
    return ER2Orchestrator(
        "Open the drawer and close it",
        [
            ER2Skill("open_drawer", "Open the drawer", "The drawer is open."),
            ER2Skill("close_drawer", "Close the drawer", "The drawer is closed."),
        ],
        client=client or _FakeClient(),
    )


def test_plan_and_observe_send_only_catalogued_decisions_and_ack_previous_call():
    client = _FakeClient()
    orchestrator = _orchestrator(client)
    image = encode_cam_high_jpeg({"images": {"cam_high": np.zeros((16, 16, 3), dtype=np.uint8)}})

    first = orchestrator.plan(image)
    second = orchestrator.observe(image, current_skill_id=first.skill_id, observation_time_seconds=2)

    assert first.decision == "set_skill"
    assert first.skill_id == "open_drawer"
    assert second.decision == "done"
    assert client.interactions.calls[1]["previous_interaction_id"] == "interaction-1"
    assert client.interactions.calls[1]["input"][0]["type"] == "function_result"
    assert client.interactions.calls[1]["input"][1]["type"] == "user_input"
    encoded = client.interactions.calls[0]["input"][0]["content"][0]["data"]
    assert base64.b64decode(encoded) == image


def test_unknown_skill_and_future_completion_are_rejected():
    with pytest.raises(ValueError, match="outside the catalog"):
        _orchestrator(_FakeClient())._parse_response(
            _FakeResponse(
                "bad-skill",
                {
                    "decision": "set_skill",
                    "skill_id": "invented_skill",
                    "previous_skill_status": "running",
                    "completion_time_seconds": 1,
                    "evidence": "unsupported",
                    "remaining_plan": [],
                    "goal_status": "in_progress",
                },
            ),
            None,
        )
    with pytest.raises(ValueError, match="later than the supplied observation"):
        _orchestrator(_FakeClient())._parse_response(
            _FakeResponse(
                "future-time",
                {
                    "decision": "set_skill",
                    "skill_id": "open_drawer",
                    "previous_skill_status": "running",
                    "completion_time_seconds": 9,
                    "evidence": "unsupported",
                    "remaining_plan": [],
                    "goal_status": "in_progress",
                },
            ),
            2,
        )


def test_latest_observation_buffer_is_latest_only():
    buffer = LatestObservationBuffer()
    first = {"images": {"cam_high": np.zeros((2, 2, 3), dtype=np.uint8)}}
    second = {"images": {"cam_high": np.ones((2, 2, 3), dtype=np.uint8)}}
    assert buffer.publish(first) == 1
    assert buffer.publish(second) == 2
    generation, observation = buffer.wait_next(0, timeout=0)
    assert generation == 2
    assert np.all(observation["images"]["cam_high"] == 1)


def test_from_config_requires_a_closed_catalog():
    with pytest.raises(ValueError, match="at least one skill"):
        ER2Orchestrator.from_config({"task": "test", "skills": []})
