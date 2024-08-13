import os

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType

from meli_forecast.params import (
    IngestionParams,
    Params,
    read_config,
)
from meli_forecast.schemas import GeoSchema, SalesSchema
from meli_forecast.utils import (
    set_mlflow_experiment,
    write_delta_table,
)


class IngestionTask:
    def __init__(self, params: IngestionParams):
        self.params = params

    def read(
        self, spark: SparkSession, file_name: str, schema: StructType
    ) -> DataFrame:
        path = os.path.join(self.params.dir, file_name)
        base_path = os.getenv("WORKSPACE_FILE_PATH")
        if base_path:
            path = "file:{base_path}/{relative_path}".format(
                base_path=base_path, relative_path=path
            )
        df = spark.read.csv(
            path, sep=self.params.sep, header=True, schema=schema
        )
        return df

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        df_sales = self.read(spark, "product_sales.csv", SalesSchema)
        df_geo = self.read(spark, "geo.csv", GeoSchema)
        write_delta_table(
            spark,
            df_sales,
            SalesSchema,
            self.params.database,
            "sales",
        )
        write_delta_table(
            spark,
            df_geo,
            GeoSchema,
            self.params.database,
            "geo",
        )


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = IngestionTask(params.ingestion)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
