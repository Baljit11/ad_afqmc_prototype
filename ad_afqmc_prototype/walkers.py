from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from jax import lax

from .core.system import system
from .core.typing import walkers


def _natorbs(dm: jax.Array, n_occ: int) -> jax.Array:
    dm = 0.5 * (dm + jnp.conj(dm.T))
    vecs = jnp.linalg.eigh(dm)[1][:, ::-1]
    return vecs[:, :n_occ]


def init_walkers(sys: system, rdm1: jax.Array, n_walkers: int) -> walkers:
    """
    Initialize walkers from natural orbitals of a trial rdm1.
    """
    wk = (sys.walker_kind).lower()
    norb = sys.norb
    nup, ndn = sys.nup, sys.ndn

    if wk == "generalized":
        ne = nup + ndn

        if rdm1.ndim == 2:
            if rdm1.shape[0] != 2 * norb or rdm1.shape[1] != 2 * norb:
                raise ValueError(
                    "For generalized walkers, a 2D rdm1 must have shape (2*norb, 2*norb)."
                )
            w0 = _natorbs(rdm1, ne)  # (2*norb, ne)
            return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

        if rdm1.ndim != 3 or rdm1.shape[0] != 2:
            raise ValueError(
                "Expected rdm1 with shape (2, norb, norb) for generalized init from spin blocks."
            )

        natorbs_up = _natorbs(rdm1[0], nup)  # (norb, nup)
        natorbs_dn = _natorbs(rdm1[1], ndn)  # (norb, ndn)

        z_up = jnp.zeros((norb, ndn))
        z_dn = jnp.zeros((norb, nup))

        top = jnp.concatenate([natorbs_up, z_up], axis=1) + 0.0j  # (norb, ne)
        bot = jnp.concatenate([z_dn, natorbs_dn], axis=1) + 0.0j  # (norb, ne)
        w0 = jnp.concatenate([top, bot], axis=0)  # (2*norb, ne)

        return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

    if rdm1.ndim == 2:
        raise ValueError(
            "For walker_kind in {'restricted','unrestricted'}, rdm1 must be spin-block (2, norb, norb)."
        )
    if rdm1.ndim != 3 or rdm1.shape[0] != 2:
        raise ValueError("Expected rdm1 with shape (2, norb, norb).")

    dm_up, dm_dn = rdm1[0], rdm1[1]

    if wk == "restricted":
        dm_tot = dm_up + dm_dn
        natorbs = _natorbs(dm_tot, nup)  # (norb, nup)
        w0 = natorbs + 0.0j
        return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

    if wk == "unrestricted":
        natorbs_up = _natorbs(dm_up, nup) + 0.0j
        natorbs_dn = _natorbs(dm_dn, ndn) + 0.0j
        wu = jnp.broadcast_to(natorbs_up, (n_walkers, *natorbs_up.shape))
        wd = jnp.broadcast_to(natorbs_dn, (n_walkers, *natorbs_dn.shape))
        return (wu, wd)

    raise ValueError(f"unknown walker_kind: {wk}")


def is_unrestricted(w: walkers) -> bool:
    return isinstance(w, tuple) and len(w) == 2


def n_walkers(w: walkers) -> int:
    return w[0].shape[0] if is_unrestricted(w) else w.shape[0]


def _chunk_size(nw: int, n_chunks: int) -> int:
    if nw % n_chunks != 0:
        raise ValueError(f"n_walkers={nw} is not divisible into n_chunks={n_chunks}")
    return nw // n_chunks


def apply_chunked(
    w: walkers,
    apply_fn: Callable,
    n_chunks: int,
    *args,
    **kwargs,
) -> jax.Array:
    """
    Apply a single walker kernel to all walkers in sequential chunks.
    n_chunks > 1 can be used to reduce memory usage at the cost of speed,
    as chunks are processed sequentially.

    apply_fn(walker_i, *args, **kwargs) -> out_i

    where walker_i can be either:
      - jax.Array (restricted/generalized)
      - tuple[jax.Array, jax.Array] (unrestricted)
    """
    nw = n_walkers(w)
    fn = lambda wi: apply_fn(wi, *args, **kwargs)

    if n_chunks == 1:
        return jax.vmap(fn, in_axes=0)(w)

    cs = _chunk_size(nw, n_chunks)
    w_c = jax.tree_util.tree_map(lambda x: x.reshape(n_chunks, cs, *x.shape[1:]), w)

    def scanned_fun(carry, cw):
        out = jax.vmap(fn, in_axes=0)(cw)
        return carry, out

    _, outs = lax.scan(scanned_fun, None, w_c)

    return jnp.reshape(outs, (nw, *outs.shape[2:]))


def apply_chunked_prop(
    w: walkers,
    fields: jax.Array,
    prop_fn: Callable,
    n_chunks: int,
    *args,
    **kwargs,
) -> walkers:
    """
    Apply a single walker propagation kernel to all walkers in sequential chunks.
    n_chunks > 1 can be used to reduce memory usage at the cost of speed,
    as chunks are processed sequentially.

    prop_fn(walker_i, fields_i, *args, **kwargs) -> walker_i

    where walker_i can be either:
      - jax.Array (restricted/generalized)
      - tuple[jax.Array, jax.Array] (unrestricted)
    """
    nw = n_walkers(w)
    fn = lambda wi, fi: prop_fn(wi, fi, *args, **kwargs)

    if n_chunks == 1:
        return jax.vmap(fn, in_axes=(0, 0))(w, fields)

    cs = _chunk_size(nw, n_chunks)
    f_c = fields.reshape(n_chunks, cs, *fields.shape[1:])
    w_c = jax.tree_util.tree_map(lambda x: x.reshape(n_chunks, cs, *x.shape[1:]), w)

    def scanned_fun(carry, xs):
        cw, cf = xs
        out = jax.vmap(fn, in_axes=(0, 0))(cw, cf)
        return carry, out

    _, outs = lax.scan(scanned_fun, None, (w_c, f_c))

    return jax.tree_util.tree_map(lambda x: x.reshape(nw, *x.shape[2:]), outs)


def _qr(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    QR with a phase convention that makes diag(R) real nonnegative.
    """
    q, r = jnp.linalg.qr(mat, mode="reduced")
    d = jnp.diag(r)
    abs_d = jnp.abs(d)
    phase = d / jnp.where(abs_d == 0, 1.0 + 0.0j, abs_d)
    q = q * jnp.conj(phase)[None, :]
    r = phase[:, None] * r
    det_r = jnp.prod(jnp.diag(r))
    return q, det_r


def orthogonalize(
    w: walkers,
    walker_kind: str,
) -> tuple[walkers, jax.Array]:
    """
    Keeps track of normalization constants.
    """
    wk = walker_kind.lower()

    if wk == "unrestricted":
        wu, wd = w
        q_u, det_u = jax.vmap(_qr, in_axes=0)(wu)
        q_d, det_d = jax.vmap(_qr, in_axes=0)(wd)
        norm = det_u * det_d
        return (q_u, q_d), norm
    elif wk in ("restricted", "generalized"):
        q, det_r = jax.vmap(_qr, in_axes=0)(w)
        norm = det_r * det_r if wk == "restricted" else det_r
        return q, norm

    raise ValueError(f"unknown walker_kind: {walker_kind}")


def orthonormalize(w: walkers, walker_kind: str) -> walkers:
    """
    Throws away normalization constants.
    """
    w_new, _ = orthogonalize(w, walker_kind)
    return w_new


def multiply_constants(w: walkers, constants: Any) -> walkers:
    """
    Multiply walkers by constants.
    """
    if is_unrestricted(w):
        wu, wd = w
        if isinstance(constants, (tuple, list)) and len(constants) == 2:
            cu, cd = constants
            return (
                wu * cu.reshape(-1, 1, 1),
                wd * cd.reshape(-1, 1, 1),
            )
        c = jnp.asarray(constants).reshape(-1, 1, 1)
        return (wu * c, wd * c)

    c = jnp.asarray(constants).reshape(-1, 1, 1)
    return w * c


def _sr_indices(
    weights: jax.Array, zeta: jax.Array | float, n_walkers: int
) -> jax.Array:
    cw = jnp.cumsum(jnp.abs(weights))
    tot = cw[-1]
    z = tot * (jnp.arange(n_walkers) + zeta) / n_walkers
    idx = jnp.searchsorted(cw, z, side="left")
    return idx


def stochastic_reconfiguration(
    w: walkers,
    weights: jax.Array,
    zeta: jax.Array | float,
    walker_kind: str,
) -> tuple[walkers, jax.Array]:
    wk = walker_kind.lower()
    n = w[0].shape[0] if wk == "unrestricted" else w.shape[0]

    cw = jnp.cumsum(jnp.abs(weights))
    avg = cw[-1] / n
    weights_new = jnp.full((n,), avg, dtype=weights.dtype)

    idx = _sr_indices(weights, zeta, n)

    if wk == "unrestricted":
        wu, wd = w
        return (wu[idx], wd[idx]), weights_new

    if wk in ("restricted", "generalized"):
        return w[idx], weights_new

    raise ValueError(f"unknown walker_kind: {walker_kind}")
