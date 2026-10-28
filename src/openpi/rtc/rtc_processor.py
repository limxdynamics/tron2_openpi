"""Real-Time Chunking (RTC) processor.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py
and LeRobot's port:
https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/rtc/modeling_rtc.py
"""

import jax
import jax.numpy as jnp

from .rtc_config import RTCAttentionSchedule
from .rtc_config import RTCConfig


def get_prefix_weights(start, end, total, schedule):
    """Compute prefix attention weights for RTC guidance.

    Compatible with both Python ints and JAX traced values.
    Uses the same algorithm as the Kinetix reference implementation.

    With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end

    `start` (inclusive) is where the chunk starts being allowed to change.
    `end` (exclusive) is where the chunk stops paying attention to the prefix.
    If start == 0, then the entire chunk is allowed to change.
    If end == total, then the entire prefix is attended to.
    `end` takes precedence over `start`: if end < start, start is pushed to end.

    Args:
        start: Number of prefix steps with full weight (inference_delay).
        end: Steps up to which weights are non-zero (execution_horizon).
        total: Total number of timesteps in the action chunk.
        schedule: Weight schedule type. String ("zeros", "ones", "linear", "exp")
            or RTCAttentionSchedule enum.

    Returns:
        Weight array of shape (total,).
    """
    # Normalize schedule to lowercase string
    if isinstance(schedule, RTCAttentionSchedule):
        schedule = schedule.value.lower()

    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule in ("linear", "exp"):
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    Implements RTC prefix guidance: uses the leftover (unexecuted) prefix from
    the previous action chunk to guide the current denoising trajectory via
    autograd-based correction.

    Note: For JAX models (Pi0), the full VJP-based RTC correction is implemented
    directly in the model's sample_actions method (following the Kinetix pattern).
    This class provides configuration and weight computation utilities.
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

    def get_prefix_weights(self, start, end, total):
        """Compute prefix attention weights using the stored schedule.

        Args:
            start: Number of prefix steps with full weight (inference_delay).
            end: Steps up to which weights are non-zero (execution_horizon).
            total: Total number of timesteps in the action chunk.

        Returns:
            Weight array of shape (total,).
        """
        return get_prefix_weights(start, end, total, self.rtc_config.prefix_attention_schedule)

    def denoise_step(
        self,
        x_t,
        v_t,
        time,
        prev_chunk_left_over,
        inference_delay,
        execution_horizon=None,
    ):
        """Apply RTC guidance to a denoising step (simplified approach).

        This method uses stop_gradient on v_t, computing the VJP only through the
        linear mapping x_0 = x_t - time * v_t. For the full VJP approach that
        includes the model's gradient (Kinetix pattern), see Pi0.sample_actions.

        Note: openpi convention is t=1 is noise, t=0 is target.

        Args:
            x_t: Current latent/state. Shape ``(B, T, A)`` or ``(T, A)``.
            v_t: Base denoised velocity from the denoiser. Same shape as x_t.
            time: Scalar in [0, 1] indicating normalized time.
            prev_chunk_left_over: Unexecuted prefix from the previous chunk.
                Shape ``(B, T_prev, A)`` or ``(T_prev, A)``. If None, no guidance
                is applied and the method returns v_t unchanged.
            inference_delay: Number of timesteps from the prefix to use for guidance.
            execution_horizon: Horizon for prefix weights. If None, uses
                ``self.rtc_config.execution_horizon``.

        Returns:
            Guided velocity with the same shape as v_t.
        """
        if prev_chunk_left_over is None:
            return v_t

        # In the original PI implementation, time goes from 0 to 1.
        # In openpi's convention, time goes from 1 (noise) to 0 (target).
        # So we invert: tau = 1 - time
        tau = 1.0 - time

        # Handle batch dimension
        squeezed = False
        if len(x_t.shape) < 3:
            x_t = x_t[None, ...]
            v_t = v_t[None, ...]
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            prev_chunk_left_over = prev_chunk_left_over[None, ...]

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # If prev chunk is shorter than execution horizon, clamp
        execution_horizon = min(execution_horizon, prev_chunk_left_over.shape[1])

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        # Pad prev_chunk_left_over if shorter than current chunk or action dim
        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = jnp.zeros((batch_size, action_chunk_size, action_dim))
            padded = padded.at[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]].set(
                prev_chunk_left_over
            )
            prev_chunk_left_over = padded

        # Compute prefix weights
        weights = get_prefix_weights(
            inference_delay, execution_horizon, action_chunk_size, self.rtc_config.prefix_attention_schedule
        )[None, :, None]

        # RTC correction using jax.vjp (simplified: stop_gradient on v_t)
        v_t_sg = jax.lax.stop_gradient(v_t)

        def x1_t_fn(x_t_arg):
            return x_t_arg - time * v_t_sg

        x1_t = x1_t_fn(x_t)
        err = (prev_chunk_left_over - x1_t) * weights

        # Compute VJP: correction = (dx1_t/dx_t)^T . err
        _, vjp_fn = jax.vjp(x1_t_fn, x_t)
        correction = vjp_fn(err)[0]

        # Compute guidance weight
        one_minus_tau = 1.0 - tau
        squared_one_minus_tau = one_minus_tau**2
        inv_r2 = (squared_one_minus_tau + tau**2) / (squared_one_minus_tau + 1e-8)
        c = jnp.where(tau > 1e-8, one_minus_tau / tau, self.rtc_config.max_guidance_weight)
        guidance_weight = jnp.minimum(c * inv_r2, self.rtc_config.max_guidance_weight)

        result = v_t - guidance_weight * correction

        # Remove batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)

        return result
