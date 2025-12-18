from ad_afqmc_prototype import config

config.setup_jax()

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import system
from ad_afqmc_prototype.ham.chol import ham_chol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.rhf import make_rhf_meas_ops
from ad_afqmc_prototype.trial.rhf import make_rhf_trial_ops, rhf_trial


def _rand_orthonormal_cols(key, nrow, ncol, dtype=jnp.complex128):
    """
    Random (nrow, ncol) matrix with orthonormal columns via QR.
    """
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(
        k1, (nrow, ncol), dtype=jnp.float64
    ) + 1.0j * jax.random.normal(k2, (nrow, ncol), dtype=jnp.float64)
    q, _ = jnp.linalg.qr(a, mode="reduced")
    return q.astype(dtype)


def _make_random_ham_chol(key, norb, n_chol, dtype=jnp.float64) -> ham_chol:
    """
    Build a small 'restricted' ham_chol with:
      - symmetric real h1
      - symmetric real chol[g]
    """
    k1, k2, k3 = jax.random.split(key, 3)

    a = jax.random.normal(k1, (norb, norb), dtype=dtype)
    h1 = 0.5 * (a + a.T)

    b = jax.random.normal(k2, (n_chol, norb, norb), dtype=dtype)
    chol = 0.5 * (b + jnp.swapaxes(b, 1, 2))

    h0 = jax.random.normal(k3, (), dtype=dtype)

    return ham_chol(basis="restricted", h0=h0, h1=h1, chol=chol)


def _make_walkers(key, sys: system, dtype=jnp.complex128):
    norb, nocc = sys.norb, sys.nup
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        w = _rand_orthonormal_cols(key, norb, nocc, dtype=dtype)
        return w

    if wk == "unrestricted":
        k1, k2 = jax.random.split(key)
        wu = _rand_orthonormal_cols(k1, norb, nocc, dtype=dtype)
        wd = _rand_orthonormal_cols(k2, norb, nocc, dtype=dtype)
        return (wu, wd)

    if wk == "generalized":
        w = _rand_orthonormal_cols(key, 2 * norb, 2 * nocc, dtype=dtype)
        return w

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted"])
def test_auto_force_bias_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

    sys = system(norb=norb, nelec=(nocc, nocc), walker_kind=walker_kind)

    key = jax.random.PRNGKey(0)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = rhf_trial(mo_coeff=_rand_orthonormal_cols(k_trial, norb, nocc))

    t_ops = make_rhf_trial_ops(sys)
    meas_manual = make_rhf_meas_ops(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = _make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


@pytest.mark.parametrize("walker_kind", ["restricted", "unrestricted"])
def test_auto_energy_matches_manual_rhf(walker_kind):
    norb = 5
    nocc = 2
    n_chol = 7

    sys = system(norb=norb, nelec=(nocc, nocc), walker_kind=walker_kind)

    key = jax.random.PRNGKey(1)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = rhf_trial(mo_coeff=_rand_orthonormal_cols(k_trial, norb, nocc))

    t_ops = make_rhf_trial_ops(sys)
    meas_manual = make_rhf_meas_ops(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = _make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        emr = jnp.real(em)
        ear = jnp.real(ea)

        assert jnp.allclose(ear, emr, rtol=5e-3, atol=5e-4), (ear, emr)


def test_auto_force_bias_matches_manual_rhf_generalized():
    walker_kind = "generalized"
    norb = 5
    nocc = 2
    n_chol = 7

    sys = system(norb=norb, nelec=(nocc, nocc), walker_kind=walker_kind)

    key = jax.random.PRNGKey(2)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = _make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = rhf_trial(mo_coeff=_rand_orthonormal_cols(k_trial, norb, nocc))

    t_ops = make_rhf_trial_ops(sys)
    meas_manual = make_rhf_meas_ops(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = _make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)
        assert jnp.allclose(v_a, v_m, rtol=1e-7, atol=1e-8), (v_a, v_m)


if __name__ == "__main__":
    pytest.main([__file__])
