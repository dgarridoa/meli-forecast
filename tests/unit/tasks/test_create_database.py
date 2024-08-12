import os

import mlflow
import pytest
from pyspark.sql import SparkSession

from meli_forecast.params import CommonParams
from meli_forecast.tasks.create_database import CreateDataBaseTask

conf = {
    "env": "dev",
    "tz": "America/Santiago",
    "database": "dev",
}
params = CommonParams.model_validate(conf)


@pytest.fixture(scope="module", autouse=True)
def task(spark: SparkSession) -> CreateDataBaseTask:
    task = CreateDataBaseTask(params)
    task.launch(spark)
    return task


def test_mlflow_tracking_server_is_not_empty():
    experiment = mlflow.get_experiment_by_name(
        os.environ["MLFLOW_EXPERIMENT_NAME"]
    )
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert runs.empty is False


def test_create_database(spark: SparkSession):
    assert spark.catalog.databaseExists(params.database)
