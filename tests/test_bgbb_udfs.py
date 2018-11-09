import numpy as np
from lifetimes.datasets import load_donations
from lifetimes.fitters.beta_geo_beta_binom_fitter import BetaGeoBetaBinomFitter
from pytest import fixture

from bgbb import BGBB
from bgbb.sql.bgbb_udfs import mk_udfs
from bgbb.wrappers import to_abgd_od

df = fixture(load_donations)


@fixture
def dfs(df, spark):
    return spark.createDataFrame(df)


@fixture
def params():
    return [1.20, 0.75, 0.66, 2.78]


@fixture
def bgbb(params):
    return BGBB(params=params)


@fixture
def ref_model(params, df):
    reference_model = BetaGeoBetaBinomFitter()
    reference_model.data = df
    reference_model.params_ = to_abgd_od([1.20, 0.75, 0.66, 2.78])
    return reference_model


@fixture
def udfs(bgbb, params):
    p_alive, n_returns = mk_udfs(
        bgbb, params=params, return_in_next_n_days=14, alive_n_days_later=0
    )
    return p_alive, n_returns


@fixture
def p_alive(udfs):
    p_alive, _ = udfs
    return p_alive


@fixture
def n_returns(udfs):
    _, n_returns = udfs
    return n_returns


def test_p_alive(dfs, p_alive, ref_model):
    pa_ref = ref_model.conditional_probability_alive(0)

    df2 = dfs.withColumn(
        "P_alive", p_alive(dfs.frequency, dfs.recency, dfs.n)
    ).toPandas()

    assert np.allclose(df2.P_alive, pa_ref)


def test_n_returns(dfs, n_returns, ref_model):
    p14_ref = ref_model.conditional_expected_number_of_purchases_up_to_time(14)

    df2 = dfs.withColumn(
        "P14", n_returns(dfs.frequency, dfs.recency, dfs.n)
    ).toPandas()

    assert np.allclose(df2.P14, p14_ref)
