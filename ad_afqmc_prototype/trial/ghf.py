from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ..core.ops import TrialOps
from ..core.system import System


@dataclass(frozen=True)
class GhfTrial:
    """
    Generalized Hartree–Fock (GHF) trial.

    mo_coeff: shape (2*norb, nelec_total)
      - rows: spin-orbitals [up-block; down-block]
      - cols: occupied orbitals (nup+ndn)
    """

    mo_coeff: jax.Array  # (2*norb, nelec_total)

    @property
    def norb(self) -> int:
        return int(self.mo_coeff.shape[0] // 2)

    @property
    def nelec_total(self) -> int:
        return int(self.mo_coeff.shape[1])


def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)


def get_rdm1(trial_data: GhfTrial) -> jax.Array:
    """
    Return spin-block 1-RDM for use by AFQMC propagator code that expects
    (2, norb, norb) for restricted-basis Hamiltonians.

    Note: This discards spin-offdiagonal blocks in a true GHF density matrix.
    """
    c = trial_data.mo_coeff
    dm = c @ c.conj().T  # (2*norb, 2*norb)
    norb = trial_data.norb
    dm_up = dm[:norb, :norb]
    dm_dn = dm[norb:, norb:]
    return jnp.stack([dm_up, dm_dn], axis=0)  # (2, norb, norb)


def overlap_r(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Restricted walker: walker shape (norb, nocc) with nocc=nup=ndn.
    Overlap is det([C_up^H W, C_dn^H W]).
    """
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne, 2*norb)
    top = cH[:, :norb] @ walker
    bot = cH[:, norb:] @ walker
    m = jnp.hstack([top, bot])  # (ne, 2*nocc)
    return _det(m)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: GhfTrial) -> jax.Array:
    """
    Unrestricted walker: (wu, wd) with shapes (norb, nup), (norb, ndn).
    Overlap is det([C_up^H W_up, C_dn^H W_dn]).
    """
    wu, wd = walker
    norb = trial_data.norb
    cH = trial_data.mo_coeff.conj().T  # (ne, 2*norb)
    top = cH[:, :norb] @ wu
    bot = cH[:, norb:] @ wd
    m = jnp.hstack([top, bot])  # (ne, ne)
    return _det(m)


def overlap_g(walker: jax.Array, trial_data: GhfTrial) -> jax.Array:
    """
    Generalized walker: walker shape (2*norb, nelec_total).
    Overlap is det(C^H W).
    """
    m = trial_data.mo_coeff.conj().T @ walker  # (ne, ne)
    return _det(m)


def make_ghf_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        if sys.nup != sys.ndn:
            raise ValueError("restricted walkers require nup == ndn.")
        return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)

    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        return TrialOps(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
