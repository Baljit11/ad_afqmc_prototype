# meas/auto.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from ..core.ops import k_energy, k_force_bias, meas_ops, trial_ops
from ..core.system import system
from ..core.typing import trial_data
from ..ham.chol import ham_chol


@dataclass(frozen=True)
class auto_meas_ctx:
    """
    Small intermediates for auto-measurements.
    """

    h1_eff: jax.Array  # (n,n) or (ns,ns)
    eps: jax.Array  # scalar


def _v0_from_chol(chol: jax.Array) -> jax.Array:
    return 0.5 * jnp.einsum("gik,gjk->ij", chol, chol, optimize="optimal")


def build_meas_ctx(
    ham_data: ham_chol, _trial_data: trial_data, eps: float = 1.0e-4
) -> auto_meas_ctx:
    v0 = _v0_from_chol(ham_data.chol)
    h1_eff = ham_data.h1 - v0
    return auto_meas_ctx(h1_eff=h1_eff, eps=jnp.asarray(eps))


def _matmul_block_diag_if_needed(mat: jax.Array, w: jax.Array) -> jax.Array:
    n = mat.shape[0]
    if w.shape[0] == n:
        return mat @ w
    if w.shape[0] == 2 * n:
        top = mat @ w[:n, :]
        bot = mat @ w[n:, :]
        return jnp.vstack([top, bot])
    raise ValueError(f"incompatible shapes: mat {mat.shape}, walker {w.shape}")


def _lin_rot_walker_array(w: jax.Array, mat: jax.Array, x: jax.Array) -> jax.Array:
    return w + x * _matmul_block_diag_if_needed(mat, w)


def _quad_rot_walker_array(w: jax.Array, mat: jax.Array, x: jax.Array) -> jax.Array:
    mw = _matmul_block_diag_if_needed(mat, w)
    mmw = _matmul_block_diag_if_needed(mat, mw)
    return w + x * mw + 0.5 * (x * x) * mmw


def _force_bias_from_overlap_array(
    w: jax.Array,
    ham_data: ham_chol,
    overlap: Callable[[jax.Array, Any], jax.Array],
    trial_data: trial_data,
) -> jax.Array:
    """
    Force bias gamma:
      <T| chol_gamma |w> / <T|w>
    computed as d/dx_gamma <T| exp(sum x_gamma chol_gamma) |w> / <T|w>
    using vjp at x=0 to linear order in the rotated walker.
    """
    chol = ham_data.chol  # (n_fields, n, n) or (n_fields, ns, ns) depending on basis
    n_fields = chol.shape[0]

    def f(x_gamma: jax.Array) -> jax.Array:
        x_chol = jnp.einsum("gij,g->ij", chol, x_gamma, optimize="optimal")
        w1 = w + _matmul_block_diag_if_needed(x_chol, w)  # linearized exp
        return overlap(w1, trial_data)

    x0 = jnp.zeros((n_fields,), dtype=w.dtype)
    val, pullback = jax.vjp(f, x0)
    grad_x = pullback(jnp.asarray(1.0 + 0.0j, dtype=val.dtype))[0]
    return grad_x / val


def force_bias_kernel_r(
    w: jax.Array,
    ham_data: ham_chol,
    _meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _force_bias_from_overlap_array(w, ham_data, overlap, trial_data)


def force_bias_kernel_g(
    w: jax.Array,
    ham_data: ham_chol,
    _meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _force_bias_from_overlap_array(w, ham_data, overlap, trial_data)


def force_bias_kernel_u(
    w: tuple[jax.Array, jax.Array],
    ham_data: ham_chol,
    _meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
) -> jax.Array:
    wu, wd = w
    chol = ham_data.chol
    n_fields = chol.shape[0]

    def f(x_gamma: jax.Array) -> jax.Array:
        x_chol = jnp.einsum("gij,g->ij", chol, x_gamma, optimize="optimal")
        wu1 = wu + x_chol @ wu
        wd1 = wd + x_chol @ wd
        return overlap((wu1, wd1), trial_data)

    x0 = jnp.zeros((n_fields,), dtype=wu.dtype)
    val, pullback = jax.vjp(f, x0)
    grad_x = pullback(jnp.asarray(1.0 + 0.0j, dtype=val.dtype))[0]
    return grad_x / val


def _energy_from_overlap_array(
    w: jax.Array,
    ham_data: ham_chol,
    meas_ctx: auto_meas_ctx,
    overlap: Callable[[jax.Array, Any], jax.Array],
    trial_data: trial_data,
) -> jax.Array:
    """
    Local energy from overlap derivatives:
      E = ( d/dx <T|exp(x h1_eff)|w> + 1/2 * sum_g d^2/dx^2 <T|exp(x chol_g)|w> ) / <T|w> + h0
    where first derivative is AD (jvp) and second derivative is FD on the quadratic truncation.
    """
    h0 = ham_data.h0
    h1_eff = meas_ctx.h1_eff
    chol = ham_data.chol
    n_fields = chol.shape[0]
    eps = meas_ctx.eps

    # one-body derivative via jvp at x=0
    def f1(x: jax.Array) -> jax.Array:
        w1 = _lin_rot_walker_array(w, h1_eff, x)
        return overlap(w1, trial_data)

    x0 = jnp.asarray(0.0)
    ovlp0, d_ovlp = jax.jvp(f1, (x0,), (jnp.asarray(1.0, dtype=x0.dtype),))

    # two-body second derivative sum via FD on quadratic truncation
    def sum_overlap_quad(x: jax.Array) -> jax.Array:
        acc0 = jnp.zeros((), dtype=ovlp0.dtype)

        def body(acc, chol_i):
            wi = _quad_rot_walker_array(w, chol_i, x)
            return acc + overlap(wi, trial_data), None

        acc, _ = lax.scan(body, acc0, chol)
        return acc

    sum_p = sum_overlap_quad(+eps)
    sum_m = sum_overlap_quad(-eps)

    d2_sum = (
        sum_p - 2.0 * jnp.asarray(n_fields, dtype=ovlp0.dtype) * ovlp0 + sum_m
    ) / (eps * eps)

    return (d_ovlp + 0.5 * d2_sum) / ovlp0 + h0


def energy_kernel_r(
    w: jax.Array,
    ham_data: ham_chol,
    meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _energy_from_overlap_array(w, ham_data, meas_ctx, overlap, trial_data)


def energy_kernel_g(
    w: jax.Array,
    ham_data: ham_chol,
    meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
):
    return _energy_from_overlap_array(w, ham_data, meas_ctx, overlap, trial_data)


def energy_kernel_u(
    w: tuple[jax.Array, jax.Array],
    ham_data: ham_chol,
    meas_ctx: auto_meas_ctx,
    trial_data: trial_data,
    *,
    overlap,
) -> jax.Array:
    wu, wd = w
    h0 = ham_data.h0
    h1_eff = meas_ctx.h1_eff
    chol = ham_data.chol
    n_fields = chol.shape[0]
    eps = meas_ctx.eps

    # one-body derivative via jvp
    def f1(x: jax.Array) -> jax.Array:
        wu1 = wu + x * (h1_eff @ wu)
        wd1 = wd + x * (h1_eff @ wd)
        return overlap((wu1, wd1), trial_data)

    x0 = jnp.asarray(0.0)
    ovlp0, d_ovlp = jax.jvp(f1, (x0,), (jnp.asarray(1.0, dtype=x0.dtype),))

    def sum_overlap_quad(x: jax.Array) -> jax.Array:
        acc0 = jnp.zeros((), dtype=ovlp0.dtype)

        def body(acc, chol_i):
            wu1 = wu + x * (chol_i @ wu) + 0.5 * (x * x) * (chol_i @ (chol_i @ wu))
            wd1 = wd + x * (chol_i @ wd) + 0.5 * (x * x) * (chol_i @ (chol_i @ wd))
            return acc + overlap((wu1, wd1), trial_data), None

        acc, _ = lax.scan(body, acc0, chol)
        return acc

    sum_p = sum_overlap_quad(+eps)
    sum_m = sum_overlap_quad(-eps)
    d2_sum = (
        sum_p - 2.0 * jnp.asarray(n_fields, dtype=ovlp0.dtype) * ovlp0 + sum_m
    ) / (eps * eps)

    return (d_ovlp + 0.5 * d2_sum) / ovlp0 + h0


def make_auto_meas_ops(
    sys: system,
    trial_ops_: trial_ops,
    *,
    eps: float = 1.0e-4,
) -> meas_ops:
    """
    Measurement ops that compute force bias and energy by differentiating overlaps.
    This reuses the trial overlap from `trial_ops_` and avoids trial-specific
    half-rotated formulas.

    Note: build_meas_ctx does NOT depend on trial_data for this implementation,
    but we keep the signature (ham, trial) for compatibility.
    """
    wk = sys.walker_kind.lower()
    overlap = trial_ops_.overlap

    def build_ctx(ham_data: ham_chol, trial: Any) -> auto_meas_ctx:
        return build_meas_ctx(ham_data, trial, eps=eps)

    if wk == "restricted":
        return meas_ops(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={
                k_force_bias: lambda w, h, c, t: force_bias_kernel_r(
                    w, h, c, t, overlap=overlap
                ),
                k_energy: lambda w, h, c, t: energy_kernel_r(
                    w, h, c, t, overlap=overlap
                ),
            },
        )

    if wk == "unrestricted":
        return meas_ops(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={
                k_force_bias: lambda w, h, c, t: force_bias_kernel_u(
                    w, h, c, t, overlap=overlap
                ),
                k_energy: lambda w, h, c, t: energy_kernel_u(
                    w, h, c, t, overlap=overlap
                ),
            },
        )

    if wk == "generalized":
        return meas_ops(
            overlap=overlap,
            build_meas_ctx=build_ctx,
            kernels={
                k_force_bias: lambda w, h, c, t: force_bias_kernel_g(
                    w, h, c, t, overlap=overlap
                ),
                k_energy: lambda w, h, c, t: energy_kernel_g(
                    w, h, c, t, overlap=overlap
                ),
            },
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
