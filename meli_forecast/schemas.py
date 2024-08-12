from pyspark.sql.types import (
    DateType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

SalesSchema = StructType(
    [
        StructField("country", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("date", DateType(), False),
        StructField("zipcode", LongType(), False),
        StructField("sales", StringType(), True),
    ]
)

GeoSchema = StructType(
    [
        StructField("country", StringType(), False),
        StructField("s_zipcode", LongType(), False),
        StructField("e_zipcode", LongType(), False),
        StructField("city", StringType(), False),
    ]
)

InputSchema = StructType(
    [
        StructField("city", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("date", DateType(), False),
        StructField("sales", DoubleType(), False),
    ]
)

SplitSchema = StructType(
    [
        StructField("city", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("date", DateType(), False),
        StructField("sales", DoubleType(), False),
        StructField("split", StringType(), False),
    ]
)

ForecastSchema = StructType(
    [
        StructField("model", StringType(), False),
        StructField("city", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("date", DateType(), False),
        StructField("sales", DoubleType(), False),
    ]
)

MetricsSchema = StructType(
    [
        StructField("model", StringType(), False),
        StructField("city", StringType(), False),
        StructField("product_id", StringType(), False),
        StructField("metric", StringType(), False),
        StructField("value", DoubleType(), False),
    ]
)

OutputSchema = StructType(
    [
        StructField("product_id", StringType(), False),
        StructField("date", DateType(), False),
        StructField("city", StringType(), False),
        StructField("sales", DoubleType(), False),
    ]
)
