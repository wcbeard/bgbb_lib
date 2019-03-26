from typing import Optional

from bgbb import BGBB
from bgbb.bgbb_utils import AbgdParams
from pandas import Series
from pyspark.sql.functions import PandasUDFType, pandas_udf, col as C


def mk_p_alive_udf(bgbb: BGBB, params=None, alive_n_days_later=0):
    @pandas_udf("double", PandasUDFType.SCALAR)
    def p_alive(frequency, recency, n, params=params):
        arr = bgbb.cond_prob_alive_nb(
            frequency,
            recency,
            n,
            params=params,
            n_days_later=alive_n_days_later,
        )
        return Series(arr)

    return p_alive


def mk_n_returns_udf(bgbb: BGBB, params=None, return_in_next_n_days=14):
    @pandas_udf("double", PandasUDFType.SCALAR)
    def n_returns(f, r, t):
        arr = bgbb.cond_exp_rets_till_nb(return_in_next_n_days, f, r, t, params)
        return Series(arr)

    return n_returns


def mk_p_th_exp_udf(bgbb: BGBB, params: Optional[AbgdParams] = None):
    "Compute expected value of latent params p, theta"

    params = params or bgbb.Params

    @pandas_udf("array<double>")
    def p_th_est(f, r, t):
        p_est, th_est = bgbb._expec_p_th(*params, f, r, t)
        return Series([[p, th] for p, th in zip(p_est, th_est)])

    @pandas_udf("double", PandasUDFType.SCALAR)
    def p_est(p_th):
        return Series([x[0] for x in p_th])

    @pandas_udf("double", PandasUDFType.SCALAR)
    def th_est(p_th):
        return Series([x[1] for x in p_th])

    return p_th_est, p_est, th_est


def add_p_th(
    bgbb: BGBB, dfs, params=None, fcol="Frequency", rcol="Recency", ncol="N"
):
    """
    Use udfs from `mk_p_th_exp_udf` to create columns `p` and `th`
    from intermediate computed column.
    """
    p_th_est, p_est, th_est = mk_p_th_exp_udf(bgbb, params=params)
    res = (
        dfs.withColumn("p_th", p_th_est(C(fcol), C(rcol), C(ncol)))
        .withColumn("p", p_est(C("p_th")))
        .withColumn("th", th_est(C("p_th")))
        .drop("p_th")
    )
    return res
