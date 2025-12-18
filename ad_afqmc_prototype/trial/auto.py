# trial/auto.py
from __future__ import annotations

from typing import Any, Callable

import jax

from ..core.ops import trial_ops
from ..core.system import system


def make_auto_trial_ops(
    sys: system,
    *,
    overlap_r: Callable[[jax.Array, Any], jax.Array],
    overlap_u: Callable[[tuple[jax.Array, jax.Array], Any], jax.Array],
    overlap_g: Callable[[jax.Array, Any], jax.Array],
    get_rdm1: Callable[[Any], jax.Array],
) -> trial_ops:
    """
    For convenience.
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return trial_ops(overlap=overlap_r, get_rdm1=get_rdm1)

    if wk == "unrestricted":
        return trial_ops(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        return trial_ops(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
