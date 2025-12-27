from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from ..ham.hubbard import HamHubbard
from ..walkers import init_walkers
from .hubbard_cpmc_ops import (
    _build_prop_ctx,
    HubbardCpmcCtx,
    HubbardCpmcOps,
    make_hubbard_cpmc_ops,
)
from .types import PropOps, PropState, QmcParams


def init_prop_state(
    *,
    sys: System,
    n_walkers: int,
    seed: int,
    ham_data: HamHubbard,
    trial_ops: TrialOps,
    trial_data: Any,
    meas_ops: MeasOps,
    params: QmcParams,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
) -> PropState:
    """
    Initialize CPMC propagation state.
    """
    key = jax.random.PRNGKey(int(seed))
    weights = jnp.ones((n_walkers,))

    if initial_walkers is None:
        initial_walkers = init_walkers(
            sys=sys, rdm1=trial_ops.get_rdm1(trial_data), n_walkers=n_walkers
        )

    initial_walkers = jax.tree_util.tree_map(lambda x: jnp.real(x), initial_walkers)

    overlaps = wk.apply_chunked(
        initial_walkers,
        trial_ops.overlap,
        n_chunks=params.n_chunks,
        trial_data=trial_data,
    )
    overlaps = jnp.real(overlaps)

    e_est = None
    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        e_kernel = meas_ops.require_kernel(k_energy)
        e_samples = jnp.real(
            wk.apply_chunked(
                initial_walkers,
                e_kernel,
                n_chunks=params.n_chunks,
                ham_data=ham_data,
                meas_ctx=meas_ctx,
                trial_data=trial_data,
            )
        )
        e_est = jnp.mean(e_samples)
    pop_shift = e_est

    node_encounters = jnp.asarray(0)

    return PropState(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=pop_shift,
        e_estimate=e_est,
        node_encounters=node_encounters,
    )


def cpmc_step(
    state: PropState,
    *,
    params: QmcParams,
    trial_data: Any,
    meas_ops: MeasOps,
    cpmc_ops: HubbardCpmcOps,
    prop_ctx: HubbardCpmcCtx,
) -> PropState:
    """
    One CPMC step with discrete spin HS fields, implemented without fast updates.
    Requires only overlap for a single walker.
    Walkers are assumed to be unrestricted and stored as (w_up, w_dn), each (nw,n,ne).
    """
    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    n_sites = cpmc_ops.n_sites()

    uniform_rns = jax.random.uniform(subkey, (nw, n_sites))

    w_floor = float(getattr(params, "weight_floor", 1.0e-8))
    w_cap = float(getattr(params, "weight_cap", 100.0))
    damping = float(getattr(params, "pop_control_damping", 0.1))

    # one body half step
    walkers = wk.apply_chunked(
        state.walkers, cpmc_ops.apply_one_body_half, params.n_chunks, prop_ctx
    )
    overlaps = wk.apply_chunked(
        walkers, meas_ops.overlap, n_chunks=params.n_chunks, trial_data=trial_data
    )
    overlaps = jnp.real(overlaps)

    # constrained-path weight update via real overlap ratio
    ratio = jnp.real(overlaps / jnp.clip(state.overlaps, min=1.0e-300))
    ratio = jnp.where(ratio < w_floor, 0.0, ratio)
    node_encounters_step = jnp.sum(ratio <= 0.0)
    weights = state.weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # two-body: scan over sites
    hs = prop_ctx.hs_constant  # (2,2)

    def scanned_fun(carry, x):
        walkers, overlaps, weights, node_encounters = carry
        w_up, w_dn = walkers

        # propose field 0 update at site x
        w0_up = w_up.at[:, x, :].mul(hs[0, 0])
        w0_dn = w_dn.at[:, x, :].mul(hs[0, 1])
        ov0 = wk.apply_chunked(
            (w0_up, w0_dn), meas_ops.overlap, params.n_chunks, trial_data
        )
        ov0 = jnp.real(ov0)
        r0 = 0.5 * jnp.real(ov0 / jnp.clip(overlaps, min=1.0e-300))
        r0 = jnp.where(r0 < w_floor, 0.0, r0)
        node_encounters += jnp.sum(r0 <= 0.0)

        # propose field 1 update at site x
        w1_up = w_up.at[:, x, :].mul(hs[1, 0])
        w1_dn = w_dn.at[:, x, :].mul(hs[1, 1])
        ov1 = wk.apply_chunked(
            (w1_up, w1_dn), meas_ops.overlap, params.n_chunks, trial_data
        )
        ov1 = jnp.real(ov1)
        r1 = 0.5 * jnp.real(ov1 / jnp.clip(overlaps, min=1.0e-300))
        r1 = jnp.where(r1 < w_floor, 0.0, r1)
        node_encounters += jnp.sum(r1 <= 0.0)

        # normalize probabilities
        norm = r0 + r1 + 1.0e-13
        p0 = r0 / norm

        rns = uniform_rns[:, x]
        choose0 = rns < p0  # (nw,)

        # update walkers
        c_up = jnp.where(choose0, hs[0, 0], hs[1, 0])
        c_dn = jnp.where(choose0, hs[0, 1], hs[1, 1])

        w_up = w_up.at[:, x, :].mul(c_up[:, None])
        w_dn = w_dn.at[:, x, :].mul(c_dn[:, None])

        overlaps = jnp.where(choose0, ov0, ov1)
        weights = weights * norm

        return ((w_up, w_dn), overlaps, weights, node_encounters), x

    (walkers, overlaps, weights, node_encounters_step), _ = lax.scan(
        scanned_fun,
        (walkers, overlaps, weights, node_encounters_step),
        jnp.arange(n_sites),
    )

    # one body half step again
    walkers = cpmc_ops.apply_one_body_half(walkers, prop_ctx)
    overlaps_new = wk.apply_chunked(
        walkers, meas_ops.overlap, n_chunks=params.n_chunks, trial_data=trial_data
    )
    overlaps_new = jnp.real(overlaps_new)

    ratio = jnp.real(overlaps_new / jnp.clip(overlaps, min=1.0e-300))
    ratio = jnp.where(ratio < w_floor, 0.0, ratio)
    node_encounters_step += jnp.sum(ratio <= 0.0)
    weights = weights * ratio
    weights = jnp.where(weights > w_cap, 0.0, weights)

    # population control factor and shift update
    weights = weights * jnp.exp(prop_ctx.dt * state.pop_control_ene_shift)
    weights = jnp.where(weights > w_cap, 0.0, weights)

    avg_w = jnp.clip(jnp.mean(weights), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    node_encounters_new = state.node_encounters + node_encounters_step

    return PropState(
        walkers=walkers,
        weights=weights,
        overlaps=overlaps_new,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
    )


def make_prop_ops(ham_data: HamHubbard, walker_kind: str) -> PropOps:
    cpmc_ops = make_hubbard_cpmc_ops(ham_data, walker_kind)

    def step(
        state: PropState,
        *,
        params: QmcParams,
        ham_data: Any,
        trial_data: Any,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: HubbardCpmcCtx,
    ) -> PropState:
        return cpmc_step(
            state,
            params=params,
            trial_data=trial_data,
            meas_ops=meas_ops,
            cpmc_ops=cpmc_ops,
            prop_ctx=prop_ctx,
        )

    def build_prop_ctx(
        ham_data: HamHubbard,
        trial_data: Any,
        params: QmcParams,
    ) -> HubbardCpmcCtx:
        return _build_prop_ctx(ham_data, params.dt)

    return PropOps(
        init_prop_state=init_prop_state,
        build_prop_ctx=build_prop_ctx,
        step=step,
    )
