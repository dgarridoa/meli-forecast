import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pyspark.sql.functions as F
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import ModelParams
from meli_forecast.schemas import ForecastSchema, SplitSchema
from meli_forecast.tasks.model import ModelTask
from meli_forecast.utils import read_table, write_delta_table
from tests.utils import assert_pyspark_df_equal

conf = {
    "env": "default",
    "tz": "America/Santiago",
    "execution_date": "2024-08-08",
    "database": "default",
    "group_columns": ["city", "product_id"],
    "time_column": "date",
    "target_column": "sales",
    "model_cls": "ExponentialSmoothing",
    "model_params": {"seasonal_periods": 7},
    "test_size": 5,
    "steps": 2,
    "freq": "1D",
}
params = ModelParams.model_validate(conf)


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 7, 9) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
                "split": (
                    ["train"] * (30 - params.test_size)
                    + ["test"] * params.test_size
                ),
            }
        ),
        schema=SplitSchema,
    )
    write_delta_table(spark, df, SplitSchema, params.database, "split")


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> ModelTask:
    create_split_table(spark)
    task = ModelTask(params)
    task.launch(spark)
    return task


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_forecast_on_test(spark: SparkSession):
    df = read_table(
        spark,
        params.database,
        "forecast_on_test",
    ).filter(F.col("model") == params.model_cls)
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": params.model_cls,
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [
                    date(2024, 8, 3) + timedelta(i)
                    for i in range(params.test_size)
                ],
                "sales": map(float, range(26, 31)),
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test.drop("sales"), df.drop("sales"))


def test_all_models_forecast(spark: SparkSession):
    df = read_table(
        spark,
        params.database,
        "all_models_forecast",
    ).filter(F.col("model") == params.model_cls)
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": params.model_cls,
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [
                    date(2024, 8, 8) + timedelta(i)
                    for i in range(params.steps)
                ],
                "sales": map(float, range(31, 33)),
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test.drop("sales"), df.drop("sales"))
