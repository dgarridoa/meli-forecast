import os
from datetime import date

import mlflow
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SparkSession

from meli_forecast.params import IngestionParams
from meli_forecast.schemas import GeoSchema, SalesSchema
from meli_forecast.tasks.ingestion import IngestionTask
from meli_forecast.utils import read_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "execution_date": "2024-08-08",
    "database": "default",
    "dir": "",
    "sep": ",",
}
params = IngestionParams.model_validate(conf)


def update_conf(spark: SparkSession):
    params.dir = spark.conf.get("spark.hive.metastore.warehouse.dir") or ""


@pytest.fixture(scope="module")
def df_sales(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
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


@pytest.fixture(scope="module")
def df_geo(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame(
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


def create_sales_csv(df_sales: DataFrame) -> None:
    df_sales.write.csv(
        os.path.join(params.dir, "product_sales.csv"),
        header=True,
        sep=params.sep,
        mode="overwrite",
    )


def create_geo_csv(df_geo: DataFrame) -> None:
    df_geo.write.csv(
        os.path.join(params.dir, "geo.csv"),
        header=True,
        sep=params.sep,
        mode="overwrite",
    )


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession, df_sales: DataFrame, df_geo: DataFrame):
    update_conf(spark)
    create_sales_csv(df_sales)
    create_geo_csv(df_geo)
    task = IngestionTask(params)
    task.launch(spark)


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_sales(spark: SparkSession, df_sales: DataFrame):
    df = read_table(spark, params.database, "sales")
    assert_pyspark_df_equal(df_sales, df)


def test_geo(spark: SparkSession, df_geo: DataFrame):
    df = read_table(spark, params.database, "geo")
    assert_pyspark_df_equal(df_geo, df)
