from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Optional

import jax

from .typing import array, ham_data, trial_data


class trial_ops(NamedTuple):
    """
    Trial operations.
      - overlap: overlap for a single walker
      - get_rdm1: trial rdm1
    """

    overlap: Callable[[Any, trial_data], array]  # (walker, trial_data) -> overlap
    get_rdm1: Callable[[trial_data], array]
    greens: Optional[Callable[[Any, trial_data], Any]] = (
        None  # (walker, trial_data) -> greens function
    )


class ham_ops(NamedTuple):
    """
    Hamiltonian (would probably be helpful when adding different Hamiltonians).
    """

    n_fields: Callable[[ham_data], int]


# a generic measurement kernel signature:
# (walker, ham_static, meas_ctx, trial_data) -> array-like
meas_kernel = Callable[[Any, Any, Any, Any], jax.Array]

# usual kernel names
k_energy = "energy"
k_force_bias = "force_bias"


@dataclass(frozen=True)
class meas_ops:
    """
    Measurement ops: trial + ham estimators + optional observables.
    """

    # same as trial_ops.overlap
    overlap: Callable[[Any, Any], jax.Array]  # (walker, trial_data) -> overlap

    # intermediates for measurements
    build_meas_ctx: Callable[[ham_data, trial_data], Any] = (
        lambda ham_data, trial_data: None
    )

    # algorithm kernels (e.g. "energy", "force_bias")
    kernels: Mapping[str, meas_kernel] = field(default_factory=dict)

    # optional observables (e.g. "mixed_rdm1", "nn_corr", ...)
    observables: Mapping[str, meas_kernel] = field(default_factory=dict)

    def has_kernel(self, name: str) -> bool:
        return name in self.kernels

    def has_observable(self, name: str) -> bool:
        return name in self.observables

    def require_kernel(self, name: str) -> meas_kernel:
        try:
            return self.kernels[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.kernels.keys()))
            raise KeyError(
                f"missing required kernel '{name}'. available: [{avail}]"
            ) from e

    def require_observable(self, name: str) -> meas_kernel:
        try:
            return self.observables[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.observables.keys()))
            raise KeyError(
                f"missing requested observable '{name}'. available: [{avail}]"
            ) from e

    def available_kernels(self) -> tuple[str, ...]:
        return tuple(sorted(self.kernels.keys()))

    def available_observables(self) -> tuple[str, ...]:
        return tuple(sorted(self.observables.keys()))
