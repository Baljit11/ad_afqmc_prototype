from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import k_energy, meas_ops
from ..core.system import system
from ..ham.chol import ham_chol
from .afqmc import afqmc_step
from .chol_afqmc_ops import chol_afqmc_ctx, chol_afqmc_ops
from .types import afqmc_params, prop_state


class block_obs(NamedTuple):
    scalars: dict[str, jax.Array]


def afqmc_block(
    state: prop_state,
    *,
    sys: system,
    params: afqmc_params,
    ham_data: ham_chol,
    trial_data: Any,
    meas_ops: meas_ops,
    prop_ops: chol_afqmc_ops,
    prop_ctx: chol_afqmc_ctx,
    meas_ctx: Any,
) -> tuple[prop_state, block_obs]:
    """
    propagation + measurement
    """

    def _scan_step(carry: prop_state, _x: Any):
        carry = afqmc_step(
            carry,
            sys=sys,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            prop_ctx=prop_ctx,
            meas_ctx=meas_ctx,
        )
        return carry, None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps_new = wk.apply_chunked(
        walkers_new, meas_ops.overlap, params.n_chunks, trial_data
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps_new)

    e_kernel = meas_ops.require_kernel(k_energy)
    e_samples = wk.apply_chunked(
        state.walkers, e_kernel, params.n_chunks, ham_data, meas_ctx, trial_data
    )
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    e_samples = jnp.where(jnp.abs(e_samples - e_ref) > thresh, e_ref, e_samples)

    weights = state.weights
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples) / w_sum_safe
    e_block = jnp.where(w_sum == 0, e_ref, e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(e_block))
    state = state._replace(
        pop_control_ene_shift=(1.0 - alpha) * state.pop_control_ene_shift
        + alpha * e_block
    )

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = wk.stochastic_reconfiguration(
        state.walkers, state.weights, zeta, sys.walker_kind
    )
    overlaps_sr = wk.apply_chunked(w_sr, meas_ops.overlap, params.n_chunks, trial_data)
    state = state._replace(
        walkers=w_sr,
        weights=weights_sr,
        overlaps=overlaps_sr,
        rng_key=key,
    )

    obs = block_obs(scalars={"energy": e_block, "weight": w_sum})
    return state, obs
