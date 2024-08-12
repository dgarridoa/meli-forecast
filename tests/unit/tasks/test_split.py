import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import SplitParams
from meli_forecast.schemas import InputSchema, SplitSchema
from meli_forecast.tasks.split import SplitTask
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
    "time_delta": 727,
    "test_size": 5,
    "freq": "1D",
}
params = SplitParams.model_validate(conf)


def create_input_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 7, 9) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
            }
        ),
        schema=InputSchema,
    )
    write_delta_table(spark, df, InputSchema, params.database, "input")


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> SplitTask:
    create_input_table(spark)
    task = SplitTask(params)
    task.launch(spark)
    return task


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_split(spark: SparkSession):
    df = read_table(spark, params.database, "split")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 7, 9) + timedelta(i) for i in range(30)],
                "sales": map(float, range(1, 31)),
                "split:": (
                    ["train"] * (30 - conf["test_size"])
                    + ["test"] * conf["test_size"]
                ),
            }
        ),
        schema=SplitSchema,
    )
    assert_pyspark_df_equal(df_test, df)
