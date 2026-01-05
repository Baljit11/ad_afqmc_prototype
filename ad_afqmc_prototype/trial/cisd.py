from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdTrial:
    """
    Restricted CISD trial in an MO basis where the reference
    determinant occupies the first nocc orbitals.

    Arrays:
      ci1: (nocc, nvir)                     singles coefficients c_{i a}
      ci2: (nocc, nvir, nocc, nvir)         doubles coefficients c_{i a j b}
    """

    ci1: jax.Array
    ci2: jax.Array

    # mixed-precision knobs (kept as static PyTree aux data)
    memory_mode: Literal["low", "high"] = "low"
    mixed_real_dtype: Any = jnp.float64
    mixed_complex_dtype: Any = jnp.complex128
    mixed_real_dtype_testing: Any = jnp.float32
    mixed_complex_dtype_testing: Any = jnp.complex64

    def __post_init__(self):
        if self.memory_mode not in ("low", "high"):
            raise ValueError(
                f"memory_mode must be one of ['low', 'high'], got {self.memory_mode}"
            )

    @property
    def nocc(self) -> int:
        return int(self.ci1.shape[0])

    @property
    def nvir(self) -> int:
        return int(self.ci1.shape[1])

    @property
    def norb(self) -> int:
        return int(self.nocc + self.nvir)

    def tree_flatten(self):
        children = (self.ci1, self.ci2)
        aux = (
            self.memory_mode,
            self.mixed_real_dtype,
            self.mixed_complex_dtype,
            self.mixed_real_dtype_testing,
            self.mixed_complex_dtype_testing,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (ci1, ci2) = children
        (
            memory_mode,
            mixed_real_dtype,
            mixed_complex_dtype,
            mixed_real_dtype_testing,
            mixed_complex_dtype_testing,
        ) = aux
        return cls(
            ci1=ci1,
            ci2=ci2,
            memory_mode=memory_mode,
            mixed_real_dtype=mixed_real_dtype,
            mixed_complex_dtype=mixed_complex_dtype,
            mixed_real_dtype_testing=mixed_real_dtype_testing,
            mixed_complex_dtype_testing=mixed_complex_dtype_testing,
        )


def get_rdm1(trial_data: CisdTrial) -> jax.Array:
    # RHF
    norb, nocc = trial_data.norb, trial_data.nocc
    occ = jnp.arange(norb) < nocc
    dm = jnp.diag(occ)
    return jnp.stack([dm, dm], axis=0).astype(float)


def overlap_r(walker: jax.Array, trial_data: CisdTrial) -> jax.Array:
    ci1, ci2 = trial_data.ci1, trial_data.ci2
    nocc = trial_data.nocc

    wocc = walker[:nocc, :]  # (nocc, nocc)
    green = jnp.linalg.solve(wocc.T, walker.T)  # (nocc, norb)

    det0 = jnp.linalg.det(wocc)
    o0 = det0 * det0

    x = green[:, nocc:]  # (nocc, nvir)
    o1 = jnp.einsum("ia,ia->", ci1, x)
    o2 = 2.0 * jnp.einsum("iajb,ia,jb->", ci2, x, x) - jnp.einsum(
        "iajb,ib,ja->", ci2, x, x
    )

    return (1.0 + 2.0 * o1 + o2) * o0


def make_cisd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("Restricted CISD trial requires nup == ndn.")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD trial currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)
