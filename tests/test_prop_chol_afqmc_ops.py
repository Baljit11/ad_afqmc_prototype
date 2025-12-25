import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.ham.chol import ham_chol
from ad_afqmc_prototype.prop.chol_afqmc_ops import make_chol_afqmc_ops


def _make_small_ham(*, norb=4, n_fields=3, h0=0.0, seed=0):
    key = jax.random.PRNGKey(seed)

    a = jax.random.normal(key, (norb, norb))
    h1 = 0.1 * (a + a.T)

    key, sub = jax.random.split(key)
    chol = 0.05 * jax.random.normal(sub, (n_fields, norb, norb))

    return ham_chol(basis="restricted", h0=jnp.asarray(h0), h1=h1, chol=chol)


def test_build_prop_ctx_shapes_and_nfields():
    norb, n_fields = 5, 7
    ham = _make_small_ham(norb=norb, n_fields=n_fields, h0=0.0)

    ops = make_chol_afqmc_ops(ham, walker_kind="restricted")
    assert ops.n_fields() == n_fields

    dm = jnp.zeros((norb, norb))
    dt = 0.2
    ctx = ops.build_prop_ctx(dm, dt)

    assert ctx.mf_shifts.shape == (n_fields,)
    assert ctx.exp_h1_half.shape == (norb, norb)
    assert ctx.dt.shape == ()
    assert ctx.sqrt_dt.shape == ()
    assert ctx.h0_prop.shape == ()


if __name__ == "__main__":
    pytest.main([__file__])
