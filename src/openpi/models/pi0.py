import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.rtc.rtc_processor import get_prefix_weights
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

        # RTC config stored as Python attributes (frozen by module_jit, like self.pi05)
        # These are NOT JAX arrays — they must not pass through jax.jit as traced values.
        self.rtc_prefix_schedule = "exp"
        self.rtc_max_guidance_weight = 10.0

        # Training-time RTC: simulated delay (None = disabled)
        self.rtc_simulated_delay = config.rtc_training_simulated_delay

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
        """Embed suffix tokens (state + action + time).

        Args:
            obs: Observation.
            noisy_actions: Noisy action tokens, shape (b, H, action_dim).
            timestep: Either scalar per batch (b,) or per-token (b, H) for
                training-time RTC.

        Returns:
            Tuple of (tokens, input_mask, ar_mask, adarms_cond).
        """
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        if timestep.ndim == 1:
            # Scalar time per batch element (standard path)
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        else:
            # Per-token time (b, H) for training-time RTC
            batch_size = timestep.shape[0]
            emb_dim = self.action_in_proj.out_features
            time_flat = timestep.reshape(-1)  # (b*H,)
            time_emb_flat = posemb_sincos(time_flat, emb_dim, min_period=4e-3, max_period=4.0)  # (b*H, emb)
            time_emb = time_emb_flat.reshape(batch_size, self.action_horizon, emb_dim)  # (b, H, emb)

        if self.pi05:
            # time MLP (for adaRMS) — uses scalar time (mean of unmasked) for conditioning
            t_for_ada = timestep if timestep.ndim == 1 else jnp.mean(timestep, axis=-1)
            ada_emb = posemb_sincos(t_for_ada, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
            ada_emb = self.time_mlp_in(ada_emb)
            ada_emb = nnx.swish(ada_emb)
            ada_emb = self.time_mlp_out(ada_emb)
            ada_emb = nnx.swish(ada_emb)
            action_expert_tokens = action_tokens
            adarms_cond = ada_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            if timestep.ndim == 1:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            else:
                time_tokens = time_emb  # already (b, H, emb)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        if self.rtc_simulated_delay is not None and self.rtc_simulated_delay > 0:
            # Training-time RTC: simulate inference delay.
            # Sample delay per batch element from Unif[0, max_delay) as in the paper.
            delay_rng = jax.random.fold_in(time_rng, 1)
            max_delay = self.rtc_simulated_delay
            delay = jax.random.randint(delay_rng, batch_shape, minval=0, maxval=max_delay)  # (b,)

            # Build per-token mask: True for positions < delay ("committed" prefix)
            pos_idx = jnp.arange(self.action_horizon)[None, :]  # (1, H)
            delay_mask = pos_idx < delay[:, None]  # (b, H)

            # Per-token time: committed positions get time=0.0 (clean/target in openpi).
            # This matches Kinetix's convention where committed positions are at the
            # "solved" end of the flow: x_t = actions, time=target.
            time_per_token = jnp.where(delay_mask, jnp.zeros_like(time[:, None]), time[:, None])  # (b, H)

            # Build x_t with per-token time:
            # committed: time=0 → x_t = 0*noise + 1*actions = actions (clean)
            # non-committed: x_t = time*noise + (1-time)*actions (normal interpolation)
            time_expanded = time_per_token[..., None]  # (b, H, 1)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            # Forward pass with per-token time
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_per_token
            )
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Loss: mask out the "committed" prefix positions
            per_token_loss = jnp.square(v_t - u_t)  # (b, H, action_dim)
            per_token_loss = jnp.mean(per_token_loss, axis=-1)  # (b, H)
            loss_mask = jnp.logical_not(delay_mask).astype(jnp.float32)  # (b, H)
            # Normalize per sample (avoid div-by-zero when delay == H)
            return jnp.sum(per_token_loss * loss_mask, axis=-1) / (jnp.sum(loss_mask, axis=-1) + 1e-8)  # (b,)

        # Standard path (no training-time RTC)
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        # RTC kwargs (schedule stays a Python attribute; numeric knobs may be JAX scalars)
        prev_chunk_left_over: at.Float[at.Array, "b ah ad"] | None = None,
        prev_chunk_left_over_len: int | at.Int[at.Array, ""] | None = None,
        inference_delay: int | at.Int[at.Array, ""] = 0,
        prefix_horizon: int | at.Int[at.Array, ""] | None = None,
        max_guidance_weight: float | at.Float[at.Array, ""] | None = None,
        # Training-time RTC inference mode: use simulated delay conditioning
        # instead of VJP guidance. Faster (~1x cost) but requires model trained
        # with rtc_training_simulated_delay.
        trained_rtc_mode: bool = False,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        if prev_chunk_left_over is not None:
            if trained_rtc_mode:
                # Training-time RTC inference: condition on fixed prefix without VJP.
                # Much faster (~1x cost) but requires a model trained with
                # rtc_training_simulated_delay.
                return self._sample_actions_trained_rtc(
                    observation,
                    noise,
                    prefix_mask,
                    kv_cache,
                    num_steps,
                    dt,
                    batch_size,
                    prev_chunk_left_over,
                    inference_delay,
                )
            # Inference-time RTC: jax.lax.scan with jax.vjp-based prefix guidance.
            # prefix_attention_schedule is read from self (frozen Python constant
            # during JIT, like self.pi05). Guidance weight can be supplied as a
            # scalar request parameter without changing input shapes.
            return self._sample_actions_rtc(
                observation,
                noise,
                prefix_mask,
                kv_cache,
                num_steps,
                dt,
                batch_size,
                prev_chunk_left_over,
                prev_chunk_left_over_len,
                inference_delay,
                prefix_horizon,
                max_guidance_weight,
            )

        # Non-RTC path: original jax.lax.while_loop
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def _sample_actions_trained_rtc(
        self,
        observation: _model.Observation,
        noise: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        num_steps: int,
        dt: float,
        batch_size: int,
        prev_chunk_left_over: jax.Array,
        inference_delay: int | jax.Array,
    ) -> jax.Array:
        """Trained-RTC inference: condition on fixed prefix without VJP.

        Instead of computing expensive VJP corrections at each denoising step,
        this directly sets the first ``inference_delay`` positions to the previous
        chunk's actions with time=0 (clean). The model, having been trained with
        ``rtc_training_simulated_delay``, has learned to generate continuations
        consistent with a committed prefix.

        Cost: ~1x of standard inference (no VJP overhead).
        Requires: model trained with rtc_training_simulated_delay > 0.
        """
        # Pad prev_chunk_left_over if shorter than action_horizon
        if prev_chunk_left_over.shape[1] < self.action_horizon:
            padded = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            padded = padded.at[:, : prev_chunk_left_over.shape[1], :].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        # Build committed mask: (1, H) → positions < inference_delay are committed
        committed_mask = jnp.arange(self.action_horizon)[None, :] < inference_delay  # (1, H)

        def step(carry, _):
            x_t, time = carry

            # Fix committed positions to prev_chunk actions
            x_t = jnp.where(committed_mask[:, :, None], prev_chunk_left_over, x_t)

            # Per-token time: committed=0.0 (clean/target), rest=current loop time
            time_per_token = jnp.where(
                committed_mask,
                jnp.zeros((batch_size, self.action_horizon)),
                jnp.broadcast_to(time, (batch_size, self.action_horizon)),
            )  # (b, H)

            # Forward pass with per-token time
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_per_token
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_expanded = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_expanded, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # Update x_t: only evolve non-committed positions
            x_t_new = x_t + dt * v_t
            x_t_new = jnp.where(committed_mask[:, :, None], prev_chunk_left_over, x_t_new)

            return (x_t_new, time + dt), None

        (x_0, _), _ = jax.lax.scan(step, (noise, jnp.array(1.0)), length=num_steps)
        return x_0

    def _sample_actions_rtc(
        self,
        observation: _model.Observation,
        noise: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        num_steps: int,
        dt: float,
        batch_size: int,
        prev_chunk_left_over: jax.Array,
        prev_chunk_left_over_len: int | jax.Array | None,
        inference_delay: int | jax.Array,
        prefix_horizon: int | jax.Array | None,
        max_guidance_weight: float | jax.Array | None,
    ) -> jax.Array:
        """RTC-enabled action sampling using Kinetix VJP approach with jax.lax.scan.

        Uses jax.lax.scan instead of Python for loop to avoid unrolling the
        VJP computation N times during JIT tracing (which causes massive XLA
        graph size and slow compilation). With scan, the step body (including
        the VJP) is compiled once and reused for all steps.

        Time convention difference from Kinetix:
        - Kinetix: t=0 is noise, t=1 is target, v_t = actions - noise, dt > 0
        - openpi:  t=1 is noise, t=0 is target, v_t = noise - actions, dt < 0

        Because openpi's v_t has opposite sign from Kinetix's, the correction
        must be SUBTRACTIVE (not additive) to maintain the correct direction:
          v_t_corrected = v_t - guidance_weight * pinv_correction

        This is equivalent to Kinetix's v_t + guidance * correction after
        accounting for the sign flip in v_t.
        """
        # Read RTC config from self (frozen Python constants during JIT)
        prefix_attention_schedule = self.rtc_prefix_schedule
        if max_guidance_weight is None:
            max_guidance_weight = self.rtc_max_guidance_weight
        if prev_chunk_left_over_len is None:
            prev_chunk_left_over_len = prev_chunk_left_over.shape[1]

        # Pad prev_chunk_left_over if shorter than action_horizon
        if prev_chunk_left_over.shape[1] < self.action_horizon:
            padded = jnp.zeros((batch_size, self.action_horizon, self.action_dim))
            padded = padded.at[:, : prev_chunk_left_over.shape[1], :].set(prev_chunk_left_over)
            prev_chunk_left_over = padded

        # Clamp guidance to the real leftover length. In the paper, this length
        # is H - s at inference start. The array may be padded only to keep JAX
        # input shapes stable; padded zeros must not guide RTC.
        if prefix_horizon is None:
            prefix_horizon = self.action_horizon
        effective_horizon = jnp.minimum(prefix_horizon, prev_chunk_left_over_len)
        effective_horizon = jnp.minimum(effective_horizon, self.action_horizon)

        # Compute prefix weights once (outside loop)
        weights = get_prefix_weights(
            inference_delay, effective_horizon, self.action_horizon, prefix_attention_schedule
        )[None, :, None]  # (1, T, 1)

        def step(carry, _):
            x_t, time = carry
            expanded_time = jnp.broadcast_to(time, (batch_size,))

            # Define denoiser: full model forward pass from x_t_arg
            # jax.vjp will compute dx_0/dx_t through the entire model (Kinetix core trick)
            def denoiser(x_t_arg):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t_arg, expanded_time
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                # Denoised prediction: x_0 = x_t - time * v_t (openpi convention)
                x_0 = x_t_arg - time * v_t
                return x_0, v_t

            # jax.vjp wraps the entire denoiser, computing full VJP through the model
            x_0, vjp_fn, v_t = jax.vjp(denoiser, x_t, has_aux=True)

            # Compute error and pseudoinverse correction
            error = (prev_chunk_left_over - x_0) * weights
            pinv_correction = vjp_fn(error)[0]

            # Compute guidance weight (convert to Kinetix time convention)
            # tau = 1 - time (Kinetix: tau goes 0→1)
            tau = 1.0 - time
            one_minus_tau = time  # = 1 - tau
            inv_r2 = (one_minus_tau**2 + tau**2) / (one_minus_tau**2 + 1e-8)
            c = jnp.where(tau > 1e-8, one_minus_tau / tau, max_guidance_weight)
            guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)

            # SUBTRACTIVE correction: openpi's v_t = noise - actions (opposite sign to Kinetix)
            v_t = v_t - guidance_weight * pinv_correction

            x_t = x_t + dt * v_t
            time = time + dt

            return (x_t, time), None

        (x_0, _), _ = jax.lax.scan(step, (noise, jnp.array(1.0)), length=num_steps)
        return x_0
