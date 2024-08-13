from datetime import date

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import CommonParams
from meli_forecast.schemas import GeoSchema, InputSchema, SalesSchema
from meli_forecast.tasks.input import InputTask
from meli_forecast.utils import read_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "database": "default",
}
params = CommonParams.model_validate(conf)


def create_sales_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "country": "B",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [
                    date(2024, 8, 1),
                    date(2024, 8, 1),
                    date(2024, 8, 7),
                    date(2024, 8, 7),
                ],
                "zipcode": [86400000.0, 3121020.0, 24725060.0, 86400000.0],
                "sales": ["1.0", "xfff", np.nan, "9999"],
            }
        ),
        schema=SalesSchema,
    )
    write_delta_table(spark, df, SalesSchema, params.database, "sales")


def create_geo_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "country": ["M", "B", "B", "B"],
                "s_zipcode": [2984000, 2984000, 24003900, 86380000],
                "e_zipcode": [3355000, 3355000, 24900001, 86600001],
                "city": ["M", "B3", "B3", "B3"],
            }
        ),
        schema=GeoSchema,
    )
    write_delta_table(spark, df, GeoSchema, params.database, "geo")


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> InputTask:
    create_sales_table(spark)
    create_geo_table(spark)
    task = InputTask(params)
    task.launch(spark)
    return task


def test_input(spark: SparkSession):
    df = read_table(spark, params.database, "input")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B3",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 8, 1), date(2024, 8, 7)],
                "sales": [1.0, 9999.0],
            }
        ),
        schema=InputSchema,
    )
    assert_pyspark_df_equal(df_test, df)
