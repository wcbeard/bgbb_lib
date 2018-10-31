from pytest import fixture
from lifetimes.datasets import load_donations
import numpy as np

from bgbb import BGBB
from wrappers import unload
# from bgbb import BGBB

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
    x, tx, n = unload(df, 'frequency recency n')
    n14_orig = bg.conditional_expected_number_of_purchases_up_to_time(
        14, data=df)
    n14_nb = bg.cond_exp_rets_till_nb(14, x, tx, n, params=pars)
    n14_api = bg.rfn.cond_exp_rets_till(df, n_days_later=14)
    n14_api_nb = bg.rfn.cond_exp_rets_till(df, n_days_later=14, nb=True)
    assert np.allclose(n14_orig, n14_nb)
    assert np.allclose(n14_orig, n14_api)
    assert np.allclose(n14_orig, n14_api_nb)
