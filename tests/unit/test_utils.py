import os
from datetime import date, timedelta
from typing import cast

import pandas as pd
import pytest
from darts.timeseries import TimeSeries
from pyspark.sql import DataFrame, SparkSession

from meli_forecast.schemas import SalesSchema
from meli_forecast.utils import (
    extract_timeseries_from_pandas_dataframe,
    get_table_info,
    read_table,
    write_delta_table,
)
from tests.utils import assert_pyspark_df_equal


@pytest.fixture
def df(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
        pd.DataFrame(
            {
                "country": "B",
                "product_id": "d8371be8-234e-3289-8j64-10e658ce3002",
                "date": [date(2023, 12, 16) + timedelta(i) for i in range(30)],
                "zipcode": 1234,
                "sales": map(str, range(1, 31)),
            }
        ),
        schema=SalesSchema,
    )


def test_write_delta_table(spark: SparkSession, df: DataFrame):
    schema = SalesSchema
    database = "default"
    warehouse_dir = cast(
        str, spark.conf.get("spark.hive.metastore.warehouse.dir")
    )

    table = "internal_table"
    write_delta_table(spark, df, schema, database, table)
    assert get_table_info(spark, database, table)["Type"] == "MANAGED"
    delta_table_df = read_table(spark, database, table)
    assert_pyspark_df_equal(df, delta_table_df)

    table = "internal_table_with_partitions"
    write_delta_table(spark, df, schema, database, table, "overwrite", ["country"])
    assert get_table_info(spark, database, table)["Type"] == "MANAGED"
    delta_table_df = read_table(spark, database, table)
    assert_pyspark_df_equal(df, delta_table_df)

    table = "external_table"
    path = os.path.join(warehouse_dir, table)
    write_delta_table(spark, df, schema, database, table, path=path)
    assert get_table_info(spark, database, table)["Type"] == "EXTERNAL"
    delta_table_df = read_table(spark, database, "external_table")
    assert_pyspark_df_equal(df, delta_table_df)

    table = "external_table_with_partitions"
    path = os.path.join(warehouse_dir, table)
    write_delta_table(
        spark,
        df,
        schema,
        database,
        table,
        partition_cols=["country"],
        path=path,
    )
    assert get_table_info(spark, database, table)["Type"] == "EXTERNAL"
    delta_table_df = read_table(spark, database, table)
    assert_pyspark_df_equal(df, delta_table_df)


def test_extract_timeseries_from_pandas_dataframe():
    expected_series = pd.Series(
        range(3),
        index=pd.to_datetime(["2024-08-01", "2024-08-02", "2024-08-03"]),
    )
    expected_time_series = TimeSeries.from_series(expected_series, freq="D")

    df = pd.DataFrame(
        {
            "time": [date(2024, 8, 1), date(2024, 8, 2), date(2024, 8, 3)],
            "target": range(3),
        }
    )
    time_series = extract_timeseries_from_pandas_dataframe(
        df, "time", "target", freq="D"
    )
    assert expected_time_series == time_series
