import datetime as dt
from typing import List

import pandas as pd


def to_s3_fmt(date):
    return date.strftime("%Y%m%d")


def to_samp_ids(samp_ids: List[int]) -> str:
    """
    iter of ints to SQL string version for main_summary
    >>> to_samp_ids([0, 1, 2])
    "'0', '1', '2'"
    """
    invalid_sample = set(samp_ids) - set(range(100))
    if invalid_sample:
        raise ValueError(
            "{} is outside of the valid range [0, 99]".format(invalid_sample)
        )
    return to_sql_list(map(str, samp_ids))


def to_sql_list(xs):
    """stringify lists for SQL queries
    >>> to_sql_list([1, 2, 3]) == '1, 2, 3'
    """

    def to_sql_literal(x):
        if isinstance(x, str):
            return "'{}'".format(x)
        return str(x)

    res = ", ".join(map(to_sql_literal, xs))
    return res


def insert_country(
    q, insert_before="{sample_comment}", countries: List[str] = ["GB"]
):
    "Insert country restriction into SQL string for testing"
    i = q.find(insert_before)
    to_insert = "AND country IN ({})\n      ".format(to_sql_list(countries))
    return q[:i] + to_insert + q[i:]


def mk_time_params(HO_WIN=14, MODEL_WIN=90, ho_start="2018-08-01"):
    """
    Return container whose attributes are holdout and model input
    date ranges, specified by a training window `MODEL_WIN`,
    holdout evaluation window `HO_WIN` and holdout start date `ho_start`
    (day after last day in model window).
    """

    def r():
        pass

    r.ho_start_datet = pd.to_datetime(ho_start)
    r.ho_start_date = r.ho_start_datet.date()
    r.ho_last_date = r.ho_start_date + dt.timedelta(days=HO_WIN - 1)
    r.model_start_date = r.ho_start_date - dt.timedelta(days=MODEL_WIN)

    # Str format
    mod_ho_ev = [r.model_start_date, r.ho_start_date, r.ho_last_date]
    # mod_ho_ev_str
    (r.model_start_date_str, r.ho_start_date_str, r.ho_last_date_str) = map(
        to_s3_fmt, mod_ho_ev
    )

    # r.__dict__.update(locals())
    return r


base_query = """
with cid_day as (
    SELECT
        C.client_id
        , C.submission_date_s3
        , from_unixtime(unix_timestamp(C.submission_date_s3, 'yyyyMMdd'),
                        'yyyy-MM-dd') AS sub_date
    FROM clients_daily C
    WHERE
        app_name = 'Firefox'
        AND channel = 'release'
      {sample_comment}  AND sample_id in ({sample_ids})
)

-- clients_daily aggregates from window *before*
-- the holdout date
, cid_model as (
    SELECT
        C.client_id
        , MIN(C.sub_date) AS Min_day
        , MAX(C.sub_date) AS Max_day
        , COUNT(*) AS X
    FROM cid_day C
    WHERE
      C.submission_date_s3 >= '{model_start_date_str}'
      AND C.submission_date_s3 < '{ho_start_date_str}'
    GROUP BY 1
)

, cid_holdout as (
    SELECT
        C.client_id
        , COUNT(*) AS N_holdout
    FROM cid_day C
    WHERE
      C.submission_date_s3 >= '{ho_start_date_str}'
      AND C.submission_date_s3 <= '{ho_last_date_str}'
    GROUP BY 1
)

, rec_freq as (
    SELECT
        C.client_id
        , datediff(C.Max_day, Min_day) AS Recency
        , X - 1 AS Frequency
        -- N: # opportunities to return
        , datediff('{ho_start_date}', Min_day) - 1  AS N
        , C.Max_day
        , C.Min_day
    FROM cid_model C
)

, rec_freq_holdout as (
  SELECT R.*
        , coalesce(H.N_holdout, 0) AS N_holdout
  FROM rec_freq R
  LEFT JOIN cid_holdout H
    ON R.client_id = H.client_id
)

SELECT * FROM {qname}
"""


def mk_rec_freq_q(
    q=None,
    holdout=False,
    model_start_date_str=None,
    pcd=None,
    sample_ids="'1'",
    **k
):
    """
    holdout: pull # of returns in holdout period?
    """
    qname = "rec_freq_holdout" if holdout else "rec_freq"
    kw = dict(
        model_start_date_str=model_start_date_str,
        sample_ids=sample_ids,
        sample_comment="" if sample_ids else "--",
        qname=qname,
        pcd=pcd,
    )
    kw.update(k)
    kw = {k: v for k, v in kw.items() if v is not None}
    return q.format(**kw)


def run_rec_freq_spk(
    spark,
    rfn_base_query=base_query,
    HO_WIN=14,
    MODEL_WIN=90,
    holdout=False,
    sample_ids="'1'",
    ho_start="2018-08-01",
    ho_days_in_future=None,
):
    """
    holdout: whether to pull # of returns in holdout period. Useful
        for evaluating model.
    ho_days_in_future: int?; if 1, set the first day in holdout period
        to be tomorrow. Negative numbers will set it to before today.
    """
    if ho_days_in_future is not None:
        ho_start = dt.date.today() + dt.timedelta(days=ho_days_in_future)
    r = mk_time_params(HO_WIN=HO_WIN, MODEL_WIN=MODEL_WIN, ho_start=ho_start)
    r.q = mk_rec_freq_q(
        q=rfn_base_query,
        holdout=holdout,
        # ignore_pcd=ignore_pcd,
        sample_ids=sample_ids,
        model_start_date_str=r.model_start_date_str,
        pcd=r.model_start_date,
        ho_start_date=r.ho_start_date,
        ho_last_date_str=r.ho_last_date_str,
        ho_start_date_str=r.ho_start_date_str,
    )
    #     print(r.q)
    dfs = spark.sql(r.q)
    return dfs, r.q


def reduce_rec_freq_spk(dfs, rfn_cols=["Recency", "Frequency", "N"]):
    """Reduce r/f/n spark dataframe to r/f/n pattern count.
    This can be used for fitting.
    """
    return dfs.groupby(rfn_cols).count().withColumnRenamed("count", "n_users")


def rec_freq_spk2pandas(df_spk, MODEL_WIN):
    df = df_spk.toPandas()
    df = df.assign(
        Max_day=lambda x: pd.to_datetime(x.Max_day),
        Min_day=lambda x: pd.to_datetime(x.Min_day),
    )
    return df


def run_rec_freq(
    spark, HO_WIN=14, MODEL_WIN=90, sample_ids="'1'", ho_start="2018-08-01"
):
    df_spk, q = run_rec_freq_spk(
        spark,
        HO_WIN=HO_WIN,
        MODEL_WIN=MODEL_WIN,
        sample_ids=sample_ids,
        ho_start=ho_start,
    )
    df = rec_freq_spk2pandas(df_spk, MODEL_WIN)
    return df, df_spk, q
