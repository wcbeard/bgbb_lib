import pyspark
from pytest import fixture


@fixture
def spark():
    spark = pyspark.sql.SparkSession.builder.appName("test").getOrCreate()
    yield spark
    spark.stop()
