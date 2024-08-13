import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import OutputParams
from meli_forecast.schemas import ForecastSchema, InputSchema, OutputSchema
from meli_forecast.tasks.output import OutputTask
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
    "window_size": 7,
}
params = OutputParams.model_validate(conf)


def create_input_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": ["B1", "B3"],
                "product_id": [
                    "8fa69e11-c148-470d-a4f1-1a1781079435",
                    "bded4d22-a25e-4b42-b9d6-41d3b1a5f71b",
                ],
                "date": [date(2024, 7, 9), date(2024, 8, 6)],
                "sales": [5.0, 1.0],
            }
        ),
        schema=InputSchema,
    )
    write_delta_table(spark, df, InputSchema, params.database, "input")


def create_forecast_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": "NaiveModel",
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 8, 8) + timedelta(i) for i in range(9)],
                "sales": map(float, range(1, 10)),
            }
        ),
        schema=ForecastSchema,
    )
    write_delta_table(
        spark,
        df,
        ForecastSchema,
        params.database,
        "forecast",
    )


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> OutputTask:
    create_input_table(spark)
    create_forecast_table(spark)
    task = OutputTask(params)
    task.launch(spark)
    return task


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_output(spark: SparkSession):
    df = read_table(spark, params.database, "output")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 3
                + ["bded4d22-a25e-4b42-b9d6-41d3b1a5f71b"] * 3,
                "date": [
                    date(2024, 8, 8),
                    date(2024, 8, 9),
                    date(2024, 8, 10),
                ]
                * 2,
                "city": ["B1"] * 3 + ["B3"] * 3,
                "sales": [28.0, 35.0, 42.0] + [1.0] * 3,
            }
        ),
        schema=OutputSchema,
    )
    assert_pyspark_df_equal(df_test, df)
