from datetime import date, timedelta

import pandas as pd
from pyspark.sql import Row, SparkSession

from meli_forecast.schemas import InputSchema, SplitSchema
from meli_forecast.split import Split
from tests.utils import assert_pyspark_df_equal

group_columns = ["city", "product_id"]
time_column = "date"
target_column = "sales"
execution_date = date(2024, 8, 8)
time_delta = 68
test_size = 7
freq = "1D"


def test_split(spark: SparkSession):
    df_sales = spark.createDataFrame(
        [
            Row(
                city="B1",
                product_id="8fa69e11-c148-470d-a4f1-1a1781079435",
                date=date(2024, 7, 25),
                sales=1.0,
            ),
            Row(
                city="B1",
                product_id="8fa69e11-c148-470d-a4f1-1a1781079435",
                date=date(2024, 8, 6),
                sales=7.0,
            ),
        ],
        schema=InputSchema,
    )

    df_expected_split = spark.createDataFrame(
        pd.DataFrame(
            {
                "city": "B1",
                "product_id": "8fa69e11-c148-470d-a4f1-1a1781079435",
                "date": [date(2024, 7, 25) + timedelta(i) for i in range(14)],
                "sales": [1.0] + [0.0] * 11 + [7.0, 0.0],
                "split:": (["train"] * 7 + ["test"] * 7),
            }
        ),
        schema=SplitSchema,
    )

    split = Split(
        group_columns,
        time_column,
        target_column,
        execution_date,
        time_delta,
        test_size,
        freq,
    )
    df_split = split.transform(df_sales, SplitSchema)
    assert_pyspark_df_equal(df_expected_split, df_split)
