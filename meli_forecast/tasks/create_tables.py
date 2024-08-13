from pyspark.sql import SparkSession

from meli_forecast.params import (
    CommonParams,
    Params,
    read_config,
)
from meli_forecast.schemas import (
    ForecastSchema,
    GeoSchema,
    InputSchema,
    MetricsSchema,
    OutputSchema,
    SalesSchema,
    SplitSchema,
)
from meli_forecast.utils import (
    create_delta_table,
)


class CreateForecastTablesTask:
    def __init__(self, params: CommonParams) -> None:
        self.params = params

    def launch(self, spark: SparkSession):
        create_delta_table(
            spark,
            SalesSchema,
            self.params.database,
            "sales",
        )
        create_delta_table(
            spark,
            GeoSchema,
            self.params.database,
            "geo",
        )
        create_delta_table(
            spark,
            InputSchema,
            self.params.database,
            "input",
        )
        create_delta_table(
            spark,
            SplitSchema,
            self.params.database,
            "split",
        )
        create_delta_table(
            spark,
            ForecastSchema,
            self.params.database,
            "forecast_on_test",
            ["model"],
        )
        create_delta_table(
            spark,
            ForecastSchema,
            self.params.database,
            "all_models_forecast",
            ["model"],
        )
        create_delta_table(
            spark,
            ForecastSchema,
            self.params.database,
            "forecast",
        )
        create_delta_table(
            spark,
            MetricsSchema,
            self.params.database,
            "metrics",
        )
        create_delta_table(
            spark,
            MetricsSchema,
            self.params.database,
            "best_models",
        )
        create_delta_table(
            spark,
            OutputSchema,
            self.params.database,
            "output",
        )


def entrypoint():
    config = read_config()
    params = Params(**config)
    spark = SparkSession.builder.getOrCreate()  # type: ignore
    task = CreateForecastTablesTask(params.common)
    task.launch(spark)


if __name__ == "__main__":
    entrypoint()
