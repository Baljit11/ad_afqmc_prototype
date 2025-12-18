from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.ops import trial_ops
from ..core.system import system


@dataclass(frozen=True)
class rhf_trial:
    mo_coeff: jax.Array  # (norb, nocc)

    @property
    def norb(self) -> int:
        return int(self.mo_coeff.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.mo_coeff.shape[1])


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1(trial_data: rhf_trial) -> jax.Array:
    c = trial_data.mo_coeff
    dm = c @ c.conj().T  # (norb, norb)
    return jnp.stack([dm, dm], axis=0)  # (2, norb, norb)


def overlap_r(walker: jax.Array, trial_data: rhf_trial) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker  # (nocc, nocc)
    return _det(m) ** 2


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: rhf_trial) -> jax.Array:
    wu, wd = walker
    cu = trial_data.mo_coeff.conj().T @ wu  # (nocc_a, nocc_a)
    cd = trial_data.mo_coeff.conj().T @ wd  # (nocc_b, nocc_b)
    return _det(cu) * _det(cd)


def overlap_g(walker: jax.Array, trial_data: rhf_trial) -> jax.Array:
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (nocc, norb)
    top = cH @ walker[:norb, :]  # (nocc, 2*nocc)
    bot = cH @ walker[norb:, :]  # (nocc, 2*nocc)
    m = jnp.vstack([top, bot])  # (2*nocc, 2*nocc)
    return _det(m)


def make_rhf_trial_ops(sys: system) -> trial_ops:
    if sys.nup != sys.ndn:
        raise ValueError("RHF requires nelec[0] == nelec[1].")

    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return trial_ops(overlap=overlap_r, get_rdm1=get_rdm1)

    if wk == "unrestricted":
        return trial_ops(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        return trial_ops(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
