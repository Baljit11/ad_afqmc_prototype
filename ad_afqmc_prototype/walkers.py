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


def init_walkers(
    sys: system,
    rdm1: jax.Array,
    n_walkers: int,
    *,
    walker_kind: Optional[str] = None,
) -> walkers:
    """
    Initialize walkers from natural orbitals of a trial rdm1.
    """
    wk = (walker_kind or sys.walker_kind).lower()
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

        top = jnp.concatenate([natorbs_up, z_up], axis=1)  # (norb, ne)
        bot = jnp.concatenate([z_dn, natorbs_dn], axis=1)  # (norb, ne)
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
        w0 = natorbs
        return jnp.broadcast_to(w0, (n_walkers, *w0.shape))

    if wk == "unrestricted":
        natorbs_up = _natorbs(dm_up, nup)
        natorbs_dn = _natorbs(dm_dn, ndn)
        wu = jnp.broadcast_to(natorbs_up, (n_walkers, *natorbs_up.shape))
        wd = jnp.broadcast_to(natorbs_dn, (n_walkers, *natorbs_dn.shape))
        return (wu, wd)

    raise ValueError(f"unknown walker_kind: {wk}")


def is_unrestricted(w: walkers) -> bool:
    return isinstance(w, tuple) and len(w) == 2


def n_walkers(w: walkers) -> int:
    return w[0].shape[0] if is_unrestricted(w) else w.shape[0]


def _chunk_size(nw: int, n_chunks: int) -> int:
    # nw should be divisible by n_chunks
    return nw // n_chunks


def apply_chunked(
    w: walkers,
    apply_fn: Callable,
    n_chunks: int,
    *args,
    **kwargs,
) -> jax.Array:
    """
    Apply a single-walker kernel to all walkers in sequential chunks.
      - if walkers is an array: apply_fn(walker, *args, **kwargs)
      - if walkers is (up, dn): apply_fn(walker_up, walker_dn, *args, **kwargs)
    """
    nw = n_walkers(w)
    if n_chunks == 1:
        if is_unrestricted(w):
            wu, wd = w
            fn = lambda a, b: apply_fn(a, b, *args, **kwargs)
            return jax.vmap(fn, in_axes=(0, 0))(wu, wd)
        else:
            fn = lambda a: apply_fn(a, *args, **kwargs)
            return jax.vmap(fn, in_axes=0)(w)

    cs = _chunk_size(nw, n_chunks)

    if is_unrestricted(w):
        wu, wd = w
        wu_c = wu.reshape(n_chunks, cs, *wu.shape[1:])
        wd_c = wd.reshape(n_chunks, cs, *wd.shape[1:])

        def scanned_fun(carry, chunk):
            cu, cd = chunk
            fn = lambda a, b: apply_fn(a, b, *args, **kwargs)
            out = jax.vmap(fn, in_axes=(0, 0))(cu, cd)
            return carry, out

        _, outs = lax.scan(scanned_fun, None, (wu_c, wd_c))
        return outs.reshape(nw, *outs.shape[2:])

    else:
        w_c = w.reshape(n_chunks, cs, *w.shape[1:])

        def scanned_fun(carry, chunk):
            fn = lambda a: apply_fn(a, *args, **kwargs)
            out = jax.vmap(fn, in_axes=0)(chunk)
            return carry, out

        _, outs = lax.scan(scanned_fun, None, w_c)
        return outs.reshape(nw, *outs.shape[2:])


def apply_chunked_prop(
    w: walkers,
    fields: jax.Array,
    prop_fn: Callable,
    n_chunks: int,
    *args,
    **kwargs,
) -> walkers:
    """
    Apply a single-walker propagation kernel to all walkers in sequential chunks.
      - fields is batched: (n_walkers, ...)
      - if walkers is an array: prop_fn(walker, fields_i, *args, **kwargs) -> walker
      - if walkers is (up, dn): prop_fn(wu, wd, fields_i, *args, **kwargs) -> (wu, wd)
    """
    nw = n_walkers(w)
    if n_chunks == 1:
        if is_unrestricted(w):
            wu, wd = w
            fn = lambda a, b, f: prop_fn(a, b, f, *args, **kwargs)
            out_u, out_d = jax.vmap(fn, in_axes=(0, 0, 0))(wu, wd, fields)
            return (out_u, out_d)
        else:
            fn = lambda a, f: prop_fn(a, f, *args, **kwargs)
            return jax.vmap(fn, in_axes=(0, 0))(w, fields)

    cs = _chunk_size(nw, n_chunks)
    f_c = fields.reshape(n_chunks, cs, *fields.shape[1:])

    if is_unrestricted(w):
        wu, wd = w
        wu_c = wu.reshape(n_chunks, cs, *wu.shape[1:])
        wd_c = wd.reshape(n_chunks, cs, *wd.shape[1:])

        def scanned_fun_u(carry, chunk):
            cu, cd, cf = chunk
            fn = lambda a, b, f: prop_fn(a, b, f, *args, **kwargs)
            out_u, out_d = jax.vmap(fn, in_axes=(0, 0, 0))(cu, cd, cf)
            return carry, (out_u, out_d)

        _, outs = lax.scan(scanned_fun_u, None, (wu_c, wd_c, f_c))
        out_u, out_d = outs
        return (
            out_u.reshape(nw, *out_u.shape[2:]),
            out_d.reshape(nw, *out_d.shape[2:]),
        )

    else:
        w_c = w.reshape(n_chunks, cs, *w.shape[1:])

        def scanned_fun(carry, chunk):
            cw, cf = chunk
            fn = lambda a, f: prop_fn(a, f, *args, **kwargs)
            out = jax.vmap(fn, in_axes=(0, 0))(cw, cf)
            return carry, out

        _, outs = lax.scan(scanned_fun, None, (w_c, f_c))
        return outs.reshape(nw, *outs.shape[2:])


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
    Orthonormalize walkers.
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


def stochastic_reconfiguration_restricted(walkers, weights, zeta):
    nwalkers = walkers.shape[0]
    cumulative_weights = jnp.cumsum(jnp.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights = jnp.ones(nwalkers) * average_weight
    z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
    indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
    walkers = walkers[indices]
    return walkers, weights


def stochastic_reconfiguration_unrestricted(walkers, weights, zeta):
    nwalkers = walkers[0].shape[0]
    cumulative_weights = jnp.cumsum(jnp.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights = jnp.ones(nwalkers) * average_weight
    z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
    indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
    return [walkers[0][indices], walkers[1][indices]], weights


def stochastic_reconfiguration(
    w: walkers,
    weights: jax.Array,
    zeta: jax.Array | float,
    walker_kind: str,
):
    """
    Stochastic reconfiguration wrapper.
    """
    wk = walker_kind.lower()
    if wk == "unrestricted":
        new_w, new_weights = stochastic_reconfiguration_unrestricted(w, weights, zeta)
        return new_w, new_weights
    elif wk in ("restricted", "generalized"):
        new_w, new_weights = stochastic_reconfiguration_restricted(w, weights, zeta)
        return new_w, new_weights

    raise ValueError(f"unknown walker_kind: {walker_kind}")
