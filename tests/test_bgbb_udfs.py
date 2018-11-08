# conda install -c conda-forge pyspark
from lifetimes.datasets import load_donations
import pyspark
from pytest import fixture

from bgbb import BGBB
from bgbb.sql.bgbb_udfs import mk_udfs

spark = pyspark.sql.SparkSession.builder.appName("test").getOrCreate()
df = fixture(load_donations)
bgbb = fixture(lambda: BGBB(params=[1.20, 0.75, 0.66, 2.78]))
params = fixture(lambda: [1.20, 0.75, 0.66, 2.78])


def test_mk_udfs(df, bgbb, params):
    dfs = spark.createDataFrame(df)

    params = [1.20, 0.75, 0.66, 2.78]
    bgbb = BGBB(params=params)

    p_alive, n_returns = mk_udfs(
        bgbb, params=params, return_in_next_n_days=14, alive_n_days_later=0
    )
    dfs2 = (
        dfs.withColumn("P14", n_returns(dfs.frequency, dfs.recency, dfs.n))
        .withColumn("P_alive", p_alive(dfs.frequency, dfs.recency, dfs.n))
    )
    pa = [
        0.116,
        0.074,
        0.273,
        0.516,
        0.73,
        0.89,
        1.0,
        0.071,
        0.32,
        0.633,
        0.864,
        1.0,
        0.102,
        0.471,
        0.823,
        1.0,
        0.214,
        0.748,
        1.0,
        0.56,
        1.0,
        1.0,
    ]
    p14 = [
        0.168,
        0.198,
        0.726,
        1.373,
        1.942,
        2.366,
        2.659,
        0.275,
        1.239,
        2.447,
        3.343,
        3.868,
        0.516,
        2.393,
        4.181,
        5.077,
        1.348,
        4.702,
        6.286,
        4.196,
        7.495,
        8.704,
    ]

    df2 = dfs2.toPandas()
    assert (df2.P14.round(3) == p14).all()
    assert (df2.P_alive.round(3) == pa).all()
