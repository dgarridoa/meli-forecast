import os
from typing import Iterable

import mlflow
import pandas as pd
from darts.timeseries import TimeSeries
from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def extract_timeseries_from_pandas_dataframe(
    df: pd.DataFrame, time_column: str, target_column: str, freq: str = "D"
) -> TimeSeries:
    serie = pd.Series(
        df[target_column].values, index=pd.to_datetime(df[time_column])
    )
    time_serie = TimeSeries.from_series(serie, freq=freq)
    return time_serie


def read_csv(
    spark: SparkSession,
    path: str,
    sep: str,
    schema: StructType,
) -> DataFrame:
    return spark.read.csv(path, sep=sep, header=True, schema=schema)


def read_table(
    spark: SparkSession,
    database: str,
    table: str,
) -> DataFrame:
    return spark.read.table(f"{database}.{table}")


def create_delta_table(
    spark: SparkSession,
    schema: StructType,
    database: str,
    table: str,
    partition_cols: list[str] | None = None,
    path: str | None = None,
) -> None:
    delta_table_builder = (
        DeltaTable.createIfNotExists(spark)
        .tableName(f"{database}.{table}")
        .addColumns(schema)
    )

    if partition_cols:
        delta_table_builder = delta_table_builder.partitionedBy(partition_cols)

    if path:
        delta_table_builder.location(path).execute()
    else:
        delta_table_builder.execute()


def write_delta_table(
    spark: SparkSession,
    df: DataFrame,
    schema: StructType,
    database: str,
    table: str,
    mode: str = "overwrite",
    partition_cols: list[str] | None = None,
    path: str | None = None,
) -> None:
    create_delta_table(spark, schema, database, table, partition_cols, path)
    df = df.select(*schema.fieldNames())

    data_frame_writter = df.write.format("delta").mode(mode)

    if partition_cols:
        data_frame_writter = data_frame_writter.partitionBy(
            *partition_cols
        ).option("partitionOverwriteMode", "dynamic")

    data_frame_writter.saveAsTable(f"{database}.{table}")


def set_mlflow_experiment() -> None:
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    if not experiment_name:
        raise ValueError(
            "environment variable MLFLOW_EXPERIMENT_NAME is unset"
        )
    mlflow.set_experiment(experiment_name)


def remove_columns_from_schema(
    schema: StructType, columns: Iterable[str]
) -> StructType:
    return StructType([field for field in schema if field.name not in columns])


def get_table_info(
    spark: SparkSession, database: str, table: str
) -> dict[str, str]:
    table_info = spark.sql(f"DESCRIBE EXTENDED {database}.{table}")
    table_info_dict = {
        row["col_name"]: row["data_type"] for row in table_info.collect()
    }
    return table_info_dict
