import mlflow
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import DoubleType, StringType

from meli_forecast.params import CommonParams, Params, read_config
from meli_forecast.schemas import InputSchema
from meli_forecast.utils import (
    read_table,
    set_mlflow_experiment,
    write_delta_table,
)


def get_country_zipcodes(
    pdf_geo: pd.DataFrame,
) -> dict[str, list[tuple[int, int, str]]]:
    pdf_geo = pdf_geo.sort_values(["country", "s_zipcode"]).reset_index(
        drop=True
    )
    country_zipcodes = {}
    for country, group in pdf_geo.groupby("country"):
        country_zipcodes[country] = list(
            group[["s_zipcode", "e_zipcode", "city"]].itertuples(
                index=False, name=None
            )
        )
    return country_zipcodes


def find_city_by_zipcode(
    country_zipcodes: dict[str, list[tuple[int, int, str]]],
    country: str,
    zipcode: int,
) -> str | None:
    for start_zipcode, end_zipcode, city in country_zipcodes.get(country, []):
        if start_zipcode <= zipcode and end_zipcode > zipcode:
            return city
    return None


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
        country_zipcodes = get_country_zipcodes(df_geo.toPandas())
        find_city_by_zipcode_udf = F.udf(
            lambda country, zipcode: find_city_by_zipcode(
                country_zipcodes, country, zipcode
            ),
            StringType(),
        )
        df_input = (
            df_sales.filter(F.col("zipcode").isNotNull())
            .withColumn("city", find_city_by_zipcode_udf("country", "zipcode"))
            .filter(F.col("city").isNotNull())
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
