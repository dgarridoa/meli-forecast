import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DoubleType

from meli_forecast.params import CommonParams, Params, read_config
from meli_forecast.schemas import InputSchema
from meli_forecast.utils import (
    read_table,
    set_mlflow_experiment,
    write_delta_table,
)


class InputTask:
    def __init__(
        self,
        params: CommonParams,
    ) -> None:
        self.params = params

    def read(self, spark: SparkSession, table_name: str) -> DataFrame:
        df = read_table(spark, self.params.database, table_name)
        return df

    def write(self, spark: SparkSession, df: DataFrame) -> None:
        write_delta_table(
            spark,
            df,
            InputSchema,
            self.params.database,
            "input",
        )

    def transform(self, df_sales: DataFrame, df_geo: DataFrame) -> DataFrame:
        df_input = (
            df_sales.filter(F.col("zipcode").isNotNull())
            .join(df_geo, how="inner", on=["country"])
            .filter(
                (F.col("zipcode") >= F.col("s_zipcode"))
                & (F.col("zipcode") < F.col("e_zipcode"))
            )
            .withColumn("sales", F.col("sales").cast(DoubleType()))
            .fillna(0.0)
            .groupby("city", "product_id", "date")
            .agg(F.sum("sales").alias("sales"))
        )
        return df_input

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        df_sales = self.read(spark, "sales")
        df_geo = self.read(spark, "geo")
        df_input = self.transform(df_sales, df_geo)
        self.write(spark, df_input)


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = InputTask(params.common)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
