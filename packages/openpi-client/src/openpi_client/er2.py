"""Gemini Robotics ER 2 task orchestration for a language-conditioned VLA.

The adapter intentionally keeps ER 2 above the policy boundary: ER 2 returns a
validated skill decision, while the caller owns prompt switching, action queue
invalidation, and robot execution.  ``google-genai`` is imported lazily so the
base openpi-client package remains usable without the optional ER 2 extra.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from threading import Condition, Event, Lock
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

TERMINAL_DECISIONS = frozenset(("done", "failure", "clarification"))
DECISIONS = frozenset(("set_skill", "keep", "uncertain", *TERMINAL_DECISIONS))
SKILL_STATUSES = frozenset(("not_started", "running", "completed", "failed", "uncertain"))
GOAL_STATUSES = frozenset(("in_progress", "achieved", "uncertain", "failed"))


@dataclass(frozen=True)
class ER2Skill:
    """A VLA prompt that ER 2 is allowed to select."""

    skill_id: str
    prompt: str
    completion_condition: str

    def __post_init__(self) -> None:
        for field_name in ("skill_id", "prompt", "completion_condition"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"ER2 skill {field_name} must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "skill_id": self.skill_id,
            "vla_prompt": self.prompt,
            "completion_condition": self.completion_condition,
        }


@dataclass(frozen=True)
class ER2Decision:
    """One structured decision returned by ER 2."""

    decision: str
    skill_id: str
    previous_skill_status: str
    completion_time_seconds: float
    evidence: str
    remaining_plan: tuple[str, ...]
    goal_status: str

    @property
    def is_terminal(self) -> bool:
        return self.decision in TERMINAL_DECISIONS


def encode_cam_high_jpeg(observation: dict[str, Any], *, quality: int = 85) -> bytes:
    """Encode one observation's ``cam_high`` image for the ER 2 API."""

    try:
        image = np.asarray(observation["images"]["cam_high"])
    except (KeyError, TypeError) as exc:
        raise ValueError("ER2 requires observation['images']['cam_high']") from exc
    if image.ndim != 3:
        raise ValueError(f"cam_high must be an HWC image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"cam_high must have three channels, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


class LatestObservationBuffer:
    """Thread-safe latest-only observation handoff.

    The policy producer publishes observations after receiving them from the
    robot.  ER 2 consumes the newest observation without calling the robot
    environment concurrently or building an unbounded queue of stale frames.
    """

    def __init__(self) -> None:
        self._condition = Condition(Lock())
        self._generation = 0
        self._observation: dict[str, Any] | None = None

    def publish(self, observation: dict[str, Any]) -> int:
        with self._condition:
            self._observation = observation
            self._generation += 1
            self._condition.notify_all()
            return self._generation

    def wait_next(self, generation: int, timeout: float) -> tuple[int, dict[str, Any] | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._generation > generation, timeout=timeout)
            if self._generation <= generation:
                return generation, None
            return self._generation, self._observation


class ER2Orchestrator:
    """Small, stateful adapter around ``client.interactions.create``.

    The model receives the high-level task, a closed skill catalog, the latest
    camera image, and the current skill status.  The only accepted model action
    is a single ``update_execution`` function call.  Robot actions are never
    sent to Google and function acknowledgements explicitly say that physical
    success has not been verified.
    """

    def __init__(
        self,
        task: str,
        skills: list[ER2Skill],
        *,
        model: str = "gemini-robotics-er-2-preview",
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        thinking_level: str = "low",
        timeout_s: float = 90.0,
        client: Any = None,
    ) -> None:
        if not task.strip():
            raise ValueError("ER2 task must be non-empty")
        if not skills:
            raise ValueError("ER2 requires at least one skill")
        if len({skill.skill_id for skill in skills}) != len(skills):
            raise ValueError("ER2 skill_id values must be unique")
        if thinking_level not in {"low", "medium", "high"}:
            raise ValueError("ER2 thinking_level must be low, medium, or high")

        self.task = task.strip()
        self.skills = {skill.skill_id: skill for skill in skills}
        self.model = model
        self.thinking_level = thinking_level
        self.timeout_s = timeout_s
        self._interaction_id: str | None = None
        self._pending_function_result: dict[str, Any] | None = None
        self._client = client or self._create_client(api_key, api_key_env)
        self._tool = self._build_tool()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ER2Orchestrator:
        """Build an orchestrator from the public ``er2`` YAML section."""

        skills = [
            ER2Skill(
                skill_id=str(item["id"]),
                prompt=str(item["prompt"]),
                completion_condition=str(item["completion_condition"]),
            )
            for item in config.get("skills", [])
        ]
        return cls(
            task=str(config["task"]),
            skills=skills,
            model=str(config.get("model", "gemini-robotics-er-2-preview")),
            api_key_env=str(config.get("api_key_env", "GEMINI_API_KEY")),
            thinking_level=str(config.get("thinking_level", "low")),
            timeout_s=float(config.get("timeout_s", 90.0)),
        )

    def _create_client(self, api_key: str | None, api_key_env: str) -> Any:
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise ValueError(f"Set {api_key_env} or pass api_key to enable ER2")
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "ER2 requires the optional dependency google-genai; install the repository extra with "
                "`uv sync --extra er2` or `pip install 'google-genai>=2.25,<3'`."
            ) from exc
        return genai.Client(api_key=key, http_options={"timeout": int(self.timeout_s * 1000)})

    def _build_tool(self) -> dict[str, Any]:
        skill_ids = ["none", *self.skills]
        return {
            "type": "function",
            "name": "update_execution",
            "description": (
                "Register one semantic orchestration decision. This adapter does not execute a robot action "
                "and the tool result never proves physical success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": sorted(DECISIONS)},
                    "skill_id": {"type": "string", "enum": skill_ids},
                    "previous_skill_status": {"type": "string", "enum": sorted(SKILL_STATUSES)},
                    "completion_time_seconds": {
                        "type": "number",
                        "description": "Observed completion time in the supplied episode, or -1 if not observed.",
                    },
                    "evidence": {"type": "string"},
                    "remaining_plan": {"type": "array", "items": {"type": "string", "enum": list(self.skills)}},
                    "goal_status": {"type": "string", "enum": sorted(GOAL_STATUSES)},
                },
                "required": [
                    "decision",
                    "skill_id",
                    "previous_skill_status",
                    "completion_time_seconds",
                    "evidence",
                    "remaining_plan",
                    "goal_status",
                ],
            },
        }

    def _catalog_text(self) -> str:
        return json.dumps([skill.as_dict() for skill in self.skills.values()], ensure_ascii=False)

    def _initial_prompt(self) -> str:
        return (
            "You are System 2 for a language-conditioned tabletop dual-arm VLA.\n"
            "Choose one skill at a time from the closed catalog below. The VLA executes the selected prompt; "
            "you only make a semantic orchestration decision. The supplied image is the current observation, "
            "not a guarantee that any future action will happen.\n"
            "Call update_execution exactly once. Never invent a skill, arm capability, destination, or future event. "
            "A skill is complete only when its observable completion condition is visible.\n"
            f"High-level task: {self.task}\n"
            f"Closed skill catalog: {self._catalog_text()}\n"
            "Select the first skill needed from this initial image."
        )

    def _observation_prompt(
        self,
        current_skill_id: str | None,
        current_skill_status: str,
        observation_time_seconds: float | None,
    ) -> str:
        time_text = "unknown" if observation_time_seconds is None else f"{observation_time_seconds:.3f}"
        return (
            "Inspect only this latest observation and the current task history. Call update_execution exactly once. "
            "Keep the current skill if it is progressing, switch only after its completion condition is visible, "
            "and use uncertain or failure when the image does not support a safe transition. Do not treat elapsed "
            "time, a previous tool acknowledgement, or a progress percentage as proof of completion.\n"
            f"High-level task: {self.task}\n"
            f"Closed skill catalog: {self._catalog_text()}\n"
            f"Current skill: {current_skill_id or 'none'}\n"
            f"Current skill status: {current_skill_status}\n"
            f"Observation time in the replay, if known: {time_text}"
        )

    @staticmethod
    def _attribute(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _parse_response(self, response: Any, observation_time_seconds: float | None) -> tuple[ER2Decision, str]:
        calls = [
            step for step in (getattr(response, "steps", None) or [])
            if self._attribute(step, "type") == "function_call"
        ]
        if len(calls) != 1 or self._attribute(calls[0], "name") != "update_execution":
            raise ValueError("ER2 response must contain exactly one update_execution function call")
        call = calls[0]
        arguments = self._attribute(call, "arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise TypeError("ER2 function arguments must be a JSON object")

        decision = str(arguments.get("decision", ""))
        skill_id = str(arguments.get("skill_id", ""))
        previous_status = str(arguments.get("previous_skill_status", ""))
        goal_status = str(arguments.get("goal_status", ""))
        evidence = str(arguments.get("evidence", "")).strip()
        remaining_plan = tuple(str(item) for item in arguments.get("remaining_plan", []))
        completion_time = float(arguments.get("completion_time_seconds", -1.0))
        if decision not in DECISIONS:
            raise ValueError(f"ER2 returned unsupported decision {decision!r}")
        if skill_id != "none" and skill_id not in self.skills:
            raise ValueError(f"ER2 selected skill outside the catalog: {skill_id!r}")
        if previous_status not in SKILL_STATUSES:
            raise ValueError(f"ER2 returned unsupported skill status {previous_status!r}")
        if goal_status not in GOAL_STATUSES:
            raise ValueError(f"ER2 returned unsupported goal status {goal_status!r}")
        if not evidence:
            raise ValueError("ER2 evidence must be non-empty")
        if any(skill not in self.skills for skill in remaining_plan):
            raise ValueError("ER2 remaining_plan contains an unknown skill")
        if decision == "set_skill" and skill_id == "none":
            raise ValueError("set_skill requires a skill_id")
        if decision in TERMINAL_DECISIONS and skill_id != "none":
            raise ValueError("terminal ER2 decisions must use skill_id='none'")
        if observation_time_seconds is not None and completion_time > observation_time_seconds + 1e-6:
            raise ValueError("ER2 completion timestamp is later than the supplied observation")
        if completion_time < -1:
            raise ValueError("ER2 completion timestamp must be -1 or non-negative")

        interaction_id = self._attribute(response, "id")
        if not interaction_id:
            raise ValueError("ER2 response did not include an interaction id")
        return (
            ER2Decision(
                decision=decision,
                skill_id=skill_id,
                previous_skill_status=previous_status,
                completion_time_seconds=completion_time,
                evidence=evidence,
                remaining_plan=remaining_plan,
                goal_status=goal_status,
            ),
            str(interaction_id),
        )

    @staticmethod
    def _step_input(image_jpeg: bytes, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "user_input",
                "content": [
                    {"type": "image", "data": base64.b64encode(image_jpeg).decode("ascii"), "mime_type": "image/jpeg"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _call(
        self,
        image_jpeg: bytes,
        prompt: str,
        *,
        initial: bool,
        observation_time_seconds: float | None = None,
    ) -> ER2Decision:
        input_items: list[dict[str, Any]] = []
        if not initial and self._pending_function_result is not None:
            input_items.append(self._pending_function_result)
        input_items.extend(self._step_input(image_jpeg, prompt))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": [self._tool],
            "generation_config": {
                "thinking_level": self.thinking_level,
                "tool_choice": "any",
                "max_output_tokens": 1400,
            },
        }
        if self._interaction_id is not None:
            kwargs["previous_interaction_id"] = self._interaction_id
        response = self._client.interactions.create(**kwargs)
        decision, interaction_id = self._parse_response(response, observation_time_seconds)
        self._interaction_id = interaction_id
        self._pending_function_result = self._function_result(response, decision)
        return decision

    def _function_result(self, response: Any, decision: ER2Decision) -> dict[str, Any]:
        calls = [
            step for step in (getattr(response, "steps", None) or [])
            if self._attribute(step, "type") == "function_call"
        ]
        call_id = str(self._attribute(calls[0], "id"))
        payload = {
            "status": "decision_registered",
            "registered_skill_id": decision.skill_id if decision.decision == "set_skill" else "none",
            "physical_execution": False,
            "physical_success_verified": False,
        }
        return {
            "type": "function_result",
            "name": "update_execution",
            "call_id": call_id,
            "result": [{"type": "text", "text": json.dumps(payload)}],
        }

    def plan(self, image_jpeg: bytes) -> ER2Decision:
        """Choose the first VLA skill from the initial observation."""

        self.reset()
        return self._call(image_jpeg, self._initial_prompt(), initial=True)

    def observe(
        self,
        image_jpeg: bytes,
        *,
        current_skill_id: str | None,
        current_skill_status: str = "running",
        observation_time_seconds: float | None = None,
    ) -> ER2Decision:
        """Evaluate one new observation and choose keep, switch, or terminal status."""

        if current_skill_id is not None and current_skill_id not in self.skills:
            raise ValueError(f"current skill is outside the catalog: {current_skill_id!r}")
        if current_skill_status not in SKILL_STATUSES:
            raise ValueError(f"unsupported current skill status: {current_skill_status!r}")
        prompt = self._observation_prompt(current_skill_id, current_skill_status, observation_time_seconds)
        return self._call(
            image_jpeg,
            prompt,
            initial=False,
            observation_time_seconds=observation_time_seconds,
        )

    def prompt_for(self, skill_id: str) -> str:
        try:
            return self.skills[skill_id].prompt
        except KeyError as exc:
            raise ValueError(f"skill is outside the catalog: {skill_id!r}") from exc

    def reset(self) -> None:
        self._interaction_id = None
        self._pending_function_result = None

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


class ER2PromptMonitor:
    """Run ER 2 decisions from latest policy observations in a background thread."""

    def __init__(
        self,
        orchestrator: ER2Orchestrator,
        observations: LatestObservationBuffer,
        stop_event: Event,
        on_decision: Callable[[ER2Decision, ER2Orchestrator], None],
        *,
        interval_s: float = 1.0,
        jpeg_quality: int = 85,
        initial_skill_id: str | None = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("ER2 monitor interval_s must be positive")
        self.orchestrator = orchestrator
        self.observations = observations
        self.stop_event = stop_event
        self.on_decision = on_decision
        self.interval_s = interval_s
        self.jpeg_quality = jpeg_quality
        if initial_skill_id is not None and initial_skill_id not in orchestrator.skills:
            raise ValueError(f"initial skill is outside the catalog: {initial_skill_id!r}")
        self.current_skill_id = initial_skill_id
        self.current_skill_status = "running" if initial_skill_id is not None else "not_started"

    def run(self) -> None:
        generation = 0
        first = self.current_skill_id is None
        next_allowed = 0.0
        while not self.stop_event.is_set():
            wait_s = next_allowed - time.monotonic()
            if wait_s > 0:
                self.stop_event.wait(wait_s)
                if self.stop_event.is_set():
                    return
            generation, observation = self.observations.wait_next(generation, self.interval_s)
            if observation is None:
                continue
            try:
                image_jpeg = encode_cam_high_jpeg(observation, quality=self.jpeg_quality)
                if first:
                    decision = self.orchestrator.plan(image_jpeg)
                    first = False
                else:
                    decision = self.orchestrator.observe(
                        image_jpeg,
                        current_skill_id=self.current_skill_id,
                        current_skill_status=self.current_skill_status,
                    )
                self.on_decision(decision, self.orchestrator)
                if decision.decision == "set_skill":
                    self.current_skill_id = decision.skill_id
                    self.current_skill_status = "running"
                elif decision.decision == "keep":
                    self.current_skill_status = "running"
                elif decision.is_terminal:
                    self.stop_event.set()
                    return
                next_allowed = time.monotonic() + self.interval_s
            except Exception:
                logger.exception("ER2 monitor stopped after an orchestration error")
                self.stop_event.set()
                return
