from typing import List

from bgbb.sql.sql_utils import run_rec_freq_spk
from bgbb.sql.bgbb_udfs import mk_n_returns_udf, mk_p_alive_udf
from bgbb import BGBB


def extract(spark, ho_start, model_win=90, sample_ids: List[int] = []):
    "TODO: increase ho_win to evaluate model performance"
    df, q = run_rec_freq_spk(
        ho_win=1,
        model_win=model_win,
        ho_start=ho_start,
        sample_ids=sample_ids,
        spark=spark,
    )
    return df


def transform(df, bgbb_params=[0.825, 0.68, 0.0876, 1.385], return_preds=[14]):
    """
    @return_preds: for each integer value `n`, make predictions
    for how many times a client is expected to return in the next `n`
    days.
    """
    bgbb = BGBB(params=bgbb_params)

    # Create/Apply UDFs
    p_alive = mk_p_alive_udf(bgbb, params=bgbb_params, alive_n_days_later=0)
    n_returns_udfs = [
        (
            "P{}".format(days),
            mk_n_returns_udf(
                bgbb, params=bgbb_params, return_in_next_n_days=days
            ),
        )
        for days in return_preds
    ]

    df2 = df.withColumn("P_alive", p_alive(df.Frequency, df.Recency, df.N))
    for days, udf in n_returns_udfs:
        df2 = df2.withColumn(days, udf(df.Frequency, df.Recency, df.N))
    return df2
