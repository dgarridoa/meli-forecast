import mlflow
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.window import Window

from meli_forecast.params import OutputParams, Params, read_config
from meli_forecast.schemas import OutputSchema
from meli_forecast.utils import (
    read_table,
    set_mlflow_experiment,
    write_delta_table,
)


class OutputTask:
    def __init__(self, params: OutputParams):
        self.params = params

    def read(self, spark: SparkSession, table_name: str):
        return read_table(spark, self.params.database, table_name)

    def write(self, spark: SparkSession, df: DataFrame) -> None:
        write_delta_table(
            spark,
            df,
            OutputSchema,
            self.params.database,
            "output",
        )

    def get_aggregated_forecast(self, df_forecast: DataFrame):
        base_window = Window.partitionBy(*self.params.group_columns)
        forecast_window = base_window.orderBy(
            self.params.time_column
        ).rowsBetween(0, self.params.window_size - 1)
        row_window = base_window.orderBy(F.desc(self.params.time_column))
        df_agg_forecast = (
            df_forecast.withColumn(
                self.params.target_column,
                F.sum(self.params.target_column).over(forecast_window),
            )
            .withColumn("row_number", F.row_number().over(row_window))
            .filter(F.col("row_number") >= self.params.window_size)
            .drop("row_number", "model")
        )
        return df_agg_forecast

    def get_aggregated_dummy_forecast(
        self, df_input: DataFrame, df_agg_forecast: DataFrame
    ):
        df_periods = df_agg_forecast.select(self.params.time_column).distinct()
        df_groups = df_agg_forecast.select(
            *self.params.group_columns
        ).distinct()
        df_agg_dummy_forecast = (
            df_input.join(df_groups, how="anti", on=self.params.group_columns)
            .groupBy(*self.params.group_columns)
            .agg(
                F.mean(self.params.target_column).alias(
                    self.params.target_column
                )
            )
            .join(df_periods, how="cross")
        )
        return df_agg_dummy_forecast

    def get_output(
        self,
        df_aggregated_forecast: DataFrame,
        df_aggregated_dummy_forecast: DataFrame,
    ):
        df_output = df_aggregated_forecast.union(
            df_aggregated_dummy_forecast.select(
                *df_aggregated_forecast.columns
            )
        )
        return df_output

    def launch(self, spark: SparkSession):
        set_mlflow_experiment()
        with mlflow.start_run(run_name=self.__class__.__name__):
            mlflow.set_tags(self.params.__dict__)

        df_input = self.read(spark, "input")
        df_forecast = self.read(spark, "forecast")
        df_agg_forecast = self.get_aggregated_forecast(df_forecast)
        df_agg_dummy_forecast = self.get_aggregated_dummy_forecast(
            df_input, df_agg_forecast
        )
        df_output = self.get_output(df_agg_forecast, df_agg_dummy_forecast)
        self.write(spark, df_output)


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = OutputTask(params.output)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
