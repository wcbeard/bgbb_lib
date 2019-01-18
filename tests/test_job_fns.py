import datetime as dt
from functools import partial
from itertools import count

import numpy.random as nr
import pandas as pd
from pyspark.sql.types import StringType, StructField, StructType
from pytest import fixture

import bgbb.sql.job_functions as jf

MODEL_WINDOW = 90
HO_WINDOW = 10
MODEL_START = pd.to_datetime("2018-10-10")
HO_START = MODEL_START + dt.timedelta(days=MODEL_WINDOW)
HO_ENDp1 = HO_START + dt.timedelta(days=HO_WINDOW + 1)
day_range = pd.date_range(MODEL_START, HO_ENDp1)

N_CLIENTS_IN_SAMPLE = 10
S3_DAY_FMT = "%Y%m%d"


@fixture()
def create_clients_daily_table(spark, dataframe_factory):
    clientsdaily_schema = StructType(
        [
            StructField("app_name", StringType(), True),
            StructField("channel", StringType(), True),
            StructField("client_id", StringType(), True),
            StructField("sample_id", StringType(), True),
            StructField("submission_date_s3", StringType(), True),
        ]
    )

    default_sample = {
        "app_name": "Firefox",
        "channel": "release",
        "client_id": "client-id",
        "sample_id": "1",
        "submission_date_s3": "20181220",
    }

    def generate_data(dataframe_factory):
        return partial(
            dataframe_factory.create_dataframe,
            base=default_sample,
            schema=clientsdaily_schema,
        )

    def coin_flip(p):
        return nr.binomial(1, p) == 1

    def gen_coins(n_coins, abgd=[1, 3, 4, 10]):
        a, b, g, d = abgd
        p = nr.beta(a, b, size=n_coins)
        θ = nr.beta(g, d, size=n_coins)
        return p, θ

    def client_2_daily_pings(client, days):
        client_days = []
        for day in days:
            client.update(submission_date_s3=day.strftime(S3_DAY_FMT))
            client_days.append(client.copy())
        return client_days

    def gen_client_days(
        client: dict, day_range, p: float, θ: float, ensure_first=True
    ):
        """If `ensure_first`, add 1st day of day_range to their history
        so that every client will show up in `rfn`.
        """
        days_used_browser = []
        for day in day_range:
            # die coin
            if coin_flip(θ):
                break
            return_today = coin_flip(p)
            if return_today:
                days_used_browser.append(day)
        if ensure_first and not days_used_browser:
            days_used_browser = [day_range[0]]
        return client_2_daily_pings(client, days_used_browser)

    def gen_client_dicts(n_clients_in_sample, abgd=[1, 1, 1, 10]):
        samples = ["1"] * n_clients_in_sample + ["2"] * n_clients_in_sample
        ps, θs = gen_coins(abgd=abgd, n_coins=len(samples))
        ps[0], θs[0] = 1, 0  # at least someone returns every day

        cids_rows = []
        for cid, samp, p, θ in zip(count(), samples, ps, θs):
            row = default_sample.copy()
            row.update(dict(client_id=cid, sample_id=samp))

            cid_rows = gen_client_days(
                client=row, day_range=day_range, p=p, θ=θ
            )
            if not cid:
                print(cid_rows)
            cids_rows.extend(cid_rows)
        return cids_rows

    cdaily_factory = generate_data(dataframe_factory)

    # @fixture
    def gen_clients_daily(n_clients_in_sample, abgd=[1, 3, 1, 10], seed=0):
        nr.seed(seed)
        table_data = gen_client_dicts(
            n_clients_in_sample=n_clients_in_sample, abgd=abgd
        )

        dataframe = cdaily_factory(table_data)
        dataframe.createOrReplaceTempView("clients_daily")
        dataframe.cache()
        return dataframe
        # yield dataframe
        # dataframe.unpersist()

    gen_clients_daily(N_CLIENTS_IN_SAMPLE)


@fixture
def rfn(spark, create_clients_daily_table):
    create_clients_daily_table
    rfn = jf.extract(
        spark, model_win=MODEL_WINDOW, ho_start=HO_START.date(), sample_ids=[1]
    )
    rfn2 = jf.transform(rfn, return_preds=[7, 14])
    return rfn2


@fixture
def rfn_pd(rfn):
    return rfn.toPandas().set_index('client_id').sort_index()


def test_max_preds(rfn_pd):
    """
    Coins for client 1 were set for immortality:
    these should have highest predict prob(alive)
    and # of returns.
    """
    pred_cols = ["P7", "P14", "P_alive"]
    first_client_preds = rfn_pd.loc["0"][pred_cols]

    max_min = rfn_pd[pred_cols].apply(["max", "min"])

    assert (first_client_preds == max_min.loc["max"]).all()
    assert (first_client_preds > max_min.loc["min"]).all()
    assert len(rfn_pd) == N_CLIENTS_IN_SAMPLE
