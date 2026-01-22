from ad_afqmc_prototype import config

config.setup_jax()

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.uhf import make_uhf_meas_ops
from ad_afqmc_prototype.trial.uhf import UhfTrial, make_uhf_trial_ops
from ad_afqmc_prototype import testing

def _make_uhf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> UhfTrial:
    ca = testing._rand_orthonormal_cols(key, norb, nup, dtype=dtype)
    key, subkey = jax.random.split(key)
    cb = testing._rand_orthonormal_cols(subkey, norb, ndn, dtype=dtype)
    return UhfTrial(mo_coeff_a=ca, mo_coeff_b=cb)

def _make_common(
    make_trial_fn,
    make_trial_ops_fn,
    make_meas_ops_fn,
    key,
    walker_kind,
    norb,
    nup,
    ndn,
    n_chol,
    ):
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)

    key = jax.random.PRNGKey(0)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = testing._make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = make_trial_fn(k_trial, norb=norb, nup=nup, ndn=ndn)

    t_ops = make_trial_ops_fn(sys)
    meas_manual = make_meas_ops_fn(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    return sys, ham, trial, meas_manual, ctx_manual, meas_auto, ctx_auto

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("unrestricted", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(0)
    _, _, k_w = jax.random.split(key, 3)

    sys, ham, trial, meas_manual, ctx_manual, meas_auto, ctx_auto = _make_common(
        _make_uhf_trial,
        make_uhf_trial_ops,
        make_uhf_meas_ops,
        key,
        walker_kind,
        norb,
        nup,
        ndn,
        n_chol,
    )

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing._make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_uhf(walker_kind, norb, nup, ndn, n_chol):
    key = jax.random.PRNGKey(1)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    sys, ham, trial, meas_manual, ctx_manual, meas_auto, ctx_auto = _make_common(
        _make_uhf_trial,
        make_uhf_trial_ops,
        make_uhf_meas_ops,
        key,
        walker_kind,
        norb,
        nup,
        ndn,
        n_chol,
    )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing._make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)

if __name__ == "__main__":
    pytest.main([__file__])
