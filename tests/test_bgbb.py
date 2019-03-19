import numpy as np
import scipy.stats as st
from lifetimes.datasets import load_donations
from pytest import fixture

from bgbb.bgbb_utils import gen_buy_die
from bgbb.core import BGBB
from bgbb.wrappers import unload

df = fixture(load_donations)
bg = fixture(lambda: BGBB(params=[1.20, 0.75, 0.66, 2.78]))
pars = fixture(lambda: [1.20, 0.75, 0.66, 2.78])


def test_len(df):
    assert len(df) == 22


def test_cond_prob_alive_nb(df, bg, pars):
    nbres = bg.rfn.cond_prob_alive(df, params=pars, nb=True)
    res = bg.rfn.cond_prob_alive(df, params=pars, nb=False)
    assert np.allclose(nbres, res)


def test_cond_prob_exp_rets(df, bg, pars):
    x, tx, n = unload(df, "frequency recency n")
    bg.data = df
    n14_orig = bg.conditional_expected_number_of_purchases_up_to_time(14)
    n14_nb = bg.cond_exp_rets_till_nb(14, x, tx, n, params=pars)
    n14_api = bg.rfn.cond_exp_rets_till(df, n_days_later=14)
    n14_api_nb = bg.rfn.cond_exp_rets_till(df, n_days_later=14, nb=True)
    assert np.allclose(n14_orig, n14_nb)
    assert np.allclose(n14_orig, n14_api)
    assert np.allclose(n14_orig, n14_api_nb)


def gen_samps_exp(abgd, n_opps, n_users, seed=0):
    # Get median p, th from hyper params
    a, b, g, d = abgd
    actual_p = np.median(st.beta(a, b).rvs(100))
    actual_th = np.median(st.beta(g, d).rvs(100))

    pp, th = np.full(n_users, actual_p), np.full(n_users, actual_th)
    df = gen_buy_die(n_opps, n_users, abgd=abgd, p_th=[pp, th], seed=seed)
    p_est, th_est = bg._expec_p_th(*abgd, df.frequency, df.recency, df.n_opps)
    df = df.assign(P_est=p_est, Th_est=th_est)
    return actual_p, actual_th, p_est, th_est, df


def sim_expec_p_th_diffs(abgd=[70.0, 30.0, 25.0, 75.0], seed=0):
    """
    Hacky test to make sure we can recover
    params p and θ from data generated by them. Hopefully, the median
    estimated params from BGBB._expec_p_th (p_est, th_est) aren't far off
    from the actual params (p_actual, th_actual) used to generate them.
    """
    p_actual, th_actual, p_est, th_est, sim_dat = gen_samps_exp(
        abgd, n_opps=90, n_users=10000, seed=seed
    )
    p_diff = np.median(p_est) - p_actual
    assert abs(p_diff) < 0.03
    th_diff = np.median(th_est) - th_actual
    assert abs(th_diff) < 0.03
    return p_diff, th_diff


def test_expec_p_th():
    sim_expec_p_th_diffs(abgd=[25.0, 75.0, 70.0, 30.0], seed=0)
    sim_expec_p_th_diffs(abgd=[70.0, 30.0, 25.0, 75.0], seed=0)
