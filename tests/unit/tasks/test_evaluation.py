import os
from datetime import date, timedelta

import mlflow
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import EvaluationParams
from meli_forecast.schemas import ForecastSchema, MetricsSchema, SplitSchema
from meli_forecast.tasks.evaluation import EvaluationTask
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
    "metrics": ["rmse", "mae"],
    "model_selection_metric": "mae",
    "freq": "1D",
}
params = EvaluationParams.model_validate(conf)


def create_split_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 7, 29) + timedelta(i) for i in range(10)],
                "sales": map(float, range(1, 11)),
                "split": ["train"] * 8 + ["test"] * 2,
            }
        ),
        schema=SplitSchema,
    )

    write_delta_table(spark, df, SplitSchema, params.database, "split")


def create_forecast_on_test_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "city": ["B1"] * 4,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 4,
                "date": [date(2024, 8, 6), date(2024, 8, 7)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    write_delta_table(
        spark,
        df,
        ForecastSchema,
        params.database,
        "forecast_on_test",
        partition_cols=["model"],
    )


def create_all_models_forecast_table(spark: SparkSession) -> None:
    df = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "city": ["B1"] * 4,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 4,
                "date": [date(2024, 8, 8), date(2024, 8, 9)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    write_delta_table(
        spark,
        df,
        ForecastSchema,
        params.database,
        "all_models_forecast",
        partition_cols=["model"],
    )


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> EvaluationTask:
    create_split_table(spark)
    create_forecast_on_test_table(spark)
    create_all_models_forecast_table(spark)
    task = EvaluationTask(params)
    task.launch(spark)


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_metrics(spark: SparkSession):
    df = read_table(spark, params.database, "metrics")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["NaiveModel"] * 2 + ["RandomForest"] * 2,
                "city": ["B1"] * 4,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 4,
                "metric": ["rmse", "mae", "rmse", "mae"],
                "value": [1.750714, 1.75, 0.608276, 0.6],
            }
        ),
        schema=MetricsSchema,
    )
    assert_pyspark_df_equal(df_test, df, 2)


def test_best_models(spark: SparkSession):
    df = read_table(spark, params.database, "best_models")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["RandomForest"],
                "city": ["B1"],
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"],
                "metric": ["mae"],
                "value": [0.600],
            }
        ),
        schema=MetricsSchema,
    )
    assert_pyspark_df_equal(df_test, df, 2)


def test_forecast(spark: SparkSession):
    df = read_table(spark, params.database, "forecast")
    df_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["RandomForest"] * 2,
                "city": ["B1"] * 2,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 2,
                "date": [date(2024, 8, 8), date(2024, 8, 9)],
                "sales": [8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    assert_pyspark_df_equal(df_test, df)
