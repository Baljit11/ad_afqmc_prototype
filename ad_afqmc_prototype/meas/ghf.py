from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..ham.hubbard import HamHubbard
from ..trial.ghf import GhfTrial, overlap_g, overlap_r, overlap_u

# ---------------------
# chol
# ---------------------


def _green_half_u(wu: jax.Array, wd: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for unrestricted walker, returned as (nelec_total, 2*norb).
    """
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne,2n)
    top = cH[:, :norb] @ wu  # (ne,nup)
    bot = cH[:, norb:] @ wd  # (ne,ndn)
    ovlp = jnp.hstack([top, bot])  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)

    nup = wu.shape[1]
    gT = jnp.vstack([wu @ inv[:nup], wd @ inv[nup:]])  # (2n,ne)
    return gT.T  # (ne,2n)


def _green_half_r(w: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for restricted walker, returned as (nelec_total, 2*norb).
    """
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T
    top = cH[:, :norb] @ w
    bot = cH[:, norb:] @ w
    ovlp = jnp.hstack([top, bot])  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)

    nocc = w.shape[1]
    gT = jnp.vstack([w @ inv[:nocc], w @ inv[nocc:]])  # (2n,ne)
    return gT.T  # (ne,2n)


def _green_half_g(w: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Mixed half Green for generalized walker, returned as (nelec_total, 2*norb).
    """
    ovlp = trial_data.mo_coeff.conj().T @ w  # (ne,ne)
    inv = jnp.linalg.inv(ovlp)
    return (w @ inv).T  # (ne,2n)


@dataclass(frozen=True)
class GhfCholMeasCtx:
    """
    Half-rotated intermediates for GHF estimators with cholesky hamiltonian.

    rot_h1: (ne, ns) where ns = 2*norb
    rot_chol: (nchol, ne, ns)
    rot_chol_flat: (nchol, ne*ns)
    """

    rot_h1: jax.Array
    rot_chol: jax.Array
    rot_chol_flat: jax.Array


def build_meas_ctx_chol(ham_data: HamChol, trial_data: GhfTrial) -> GhfCholMeasCtx:
    cH = trial_data.mo_coeff.conj().T  # (ne, 2n)
    norb = trial_data.norb

    if ham_data.basis == "restricted":
        z = jnp.zeros_like(ham_data.h1)
        h1_so = jnp.block([[ham_data.h1, z], [z, ham_data.h1]])  # (2n,2n)
        rot_h1 = cH @ h1_so  # (ne,2n)

        chol_sp = ham_data.chol.reshape(ham_data.chol.shape[0], norb, norb)

        def _rot_one(x):
            left = cH[:, :norb] @ x
            right = cH[:, norb:] @ x
            return jnp.concatenate([left, right], axis=1)

        rot_chol = jax.vmap(_rot_one, in_axes=0)(chol_sp)  # (nchol,ne,2n)

    else:
        rot_h1 = cH @ ham_data.h1  # (ne,ns)
        rot_chol = jax.vmap(lambda x: cH @ x, in_axes=0)(ham_data.chol)  # (nchol,ne,ns)

    rot_chol_flat = rot_chol.reshape(rot_chol.shape[0], -1)
    return GhfCholMeasCtx(
        rot_h1=rot_h1, rot_chol=rot_chol, rot_chol_flat=rot_chol_flat
    )


def force_bias_kernel_from_green(
    g_half: jax.Array, meas_ctx: GhfCholMeasCtx
) -> jax.Array:
    return jnp.einsum("gij,ij->g", meas_ctx.rot_chol, g_half, optimize="optimal")


def energy_kernel_from_green(
    g_half: jax.Array, ham_data: HamChol, meas_ctx: GhfCholMeasCtx
) -> jax.Array:
    ene0 = ham_data.h0
    ene1 = jnp.sum(g_half * meas_ctx.rot_h1)
    f = jnp.einsum(
        "gij,jk->gik", meas_ctx.rot_chol, g_half.T, optimize="optimal"
    )  # (nchol,ne,ne)
    coul = jnp.trace(f, axis1=1, axis2=2)  # (nchol,)
    exc = jnp.sum(f * jnp.swapaxes(f, 1, 2))
    ene2 = 0.5 * (jnp.sum(coul * coul) - exc)

    return ene0 + ene1 + ene2


def force_bias_kernel_r(
    walker: jax.Array, ham_data: Any, meas_ctx: GhfCholMeasCtx, trial_data: GhfTrial
) -> jax.Array:
    g = _green_half_r(walker, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


def force_bias_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: Any,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    wu, wd = walker
    g = _green_half_u(wu, wd, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


def force_bias_kernel_g(
    walker: jax.Array, ham_data: Any, meas_ctx: GhfCholMeasCtx, trial_data: GhfTrial
) -> jax.Array:
    g = _green_half_g(walker, trial_data)
    return force_bias_kernel_from_green(g, meas_ctx)


def energy_kernel_r(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    g = _green_half_r(walker, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


def energy_kernel_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    wu, wd = walker
    g = _green_half_u(wu, wd, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


def energy_kernel_g(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: GhfCholMeasCtx,
    trial_data: GhfTrial,
) -> jax.Array:
    g = _green_half_g(walker, trial_data)
    return energy_kernel_from_green(g, ham_data, meas_ctx)


def make_ghf_meas_ops_chol(sys: System) -> MeasOps:
    """
    GHF measurement ops for Cholesky Hamiltonians
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return MeasOps(
            overlap=overlap_r,
            build_meas_ctx=build_meas_ctx_chol,
            kernels={k_force_bias: force_bias_kernel_r, k_energy: energy_kernel_r},
        )

    if wk == "unrestricted":
        return MeasOps(
            overlap=overlap_u,
            build_meas_ctx=build_meas_ctx_chol,
            kernels={k_force_bias: force_bias_kernel_u, k_energy: energy_kernel_u},
        )

    if wk == "generalized":
        return MeasOps(
            overlap=overlap_g,
            build_meas_ctx=build_meas_ctx_chol,
            kernels={k_force_bias: force_bias_kernel_g, k_energy: energy_kernel_g},
        )

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


# ---------------------
# hubbard
# ---------------------


def _full_green_unrestricted(
    wu: jax.Array, wd: jax.Array, trial_data: GhfTrial
) -> jax.Array:
    """
    Full Green's function (2n,2n) for unrestricted walkers.
    """
    norb = trial_data.norb
    nup = wu.shape[1]
    ndn = wd.shape[1]
    dtype = wu.dtype

    z_up = jnp.zeros((norb, ndn), dtype=dtype)
    z_dn = jnp.zeros((norb, nup), dtype=dtype)

    w_top = jnp.concatenate([wu, z_up], axis=1)  # (n, ne)
    w_bot = jnp.concatenate([z_dn, wd], axis=1)  # (n, ne)
    w_so = jnp.concatenate([w_top, w_bot], axis=0)  # (2n, ne)

    c_occ_H = trial_data.mo_coeff.conj().T  # (ne,2n)
    inv = jnp.linalg.inv(c_occ_H @ w_so)  # (ne,ne)
    g = (w_so @ inv @ c_occ_H).T  # (2n,2n)
    return g


def _full_green_generalized(w: jax.Array, trial_data: GhfTrial) -> jax.Array:
    c_occ_H = trial_data.mo_coeff.conj().T
    inv = jnp.linalg.inv(c_occ_H @ w)
    g = (w @ inv @ c_occ_H).T
    return g


def energy_kernel_hubbard_u(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    wu, wd = walker
    g = _full_green_unrestricted(wu, wd, trial_data)
    norb = trial_data.norb
    h1 = ham_data.h1
    u = ham_data.u

    e1 = jnp.sum(g[:norb, :norb] * h1) + jnp.sum(g[norb:, norb:] * h1)

    g_uu = g[:norb, :norb].diagonal()
    g_dd = g[norb:, norb:].diagonal()
    g_ud = g[:norb, norb:].diagonal()
    g_du = g[norb:, :norb].diagonal()
    e2 = u * (jnp.sum(g_uu * g_dd) - jnp.sum(g_ud * g_du))

    return e1 + e2


def energy_kernel_hubbard_g(
    walker: jax.Array,
    ham_data: HamHubbard,
    meas_ctx: Any,
    trial_data: GhfTrial,
) -> jax.Array:
    g = _full_green_generalized(walker, trial_data)
    norb = trial_data.norb
    h1 = ham_data.h1
    u = ham_data.u

    e1 = jnp.sum(g[:norb, :norb] * h1) + jnp.sum(g[norb:, norb:] * h1)

    g_uu = g[:norb, :norb].diagonal()
    g_dd = g[norb:, norb:].diagonal()
    g_ud = g[:norb, norb:].diagonal()
    g_du = g[norb:, :norb].diagonal()
    e2 = u * (jnp.sum(g_uu * g_dd) - jnp.sum(g_ud * g_du))

    return e1 + e2


def make_ghf_meas_ops_hubbard(sys: System) -> MeasOps:
    """
    GHF measurement ops for hubbard hamiltonian
    """
    wk = sys.walker_kind.lower()

    if wk == "unrestricted":
        return MeasOps(
            overlap=overlap_u,
            kernels={k_energy: energy_kernel_hubbard_u},
        )

    if wk == "generalized":
        return MeasOps(
            overlap=overlap_g,
            kernels={k_energy: energy_kernel_hubbard_g},
        )

    raise ValueError(
        f"hubbard GHF meas only implemented for unrestricted/generalized, got walker_kind={sys.walker_kind}"
    )
