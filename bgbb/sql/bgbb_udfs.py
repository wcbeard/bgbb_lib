from pyspark.sql.functions import pandas_udf, PandasUDFType
from pandas import Series


def mk_p_alive_udf(bgbb, params=None, alive_n_days_later=0):
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


def mk_n_returns_udf(bgbb, params=None, return_in_next_n_days=14):
    @pandas_udf("double", PandasUDFType.SCALAR)
    def n_returns(f, r, t):
        arr = bgbb.cond_exp_rets_till_nb(return_in_next_n_days, f, r, t, params)
        return Series(arr)
    return n_returns
