from datetime import date, timedelta

import pandas as pd
from pyspark.sql import Row, SparkSession

from meli_forecast.evaluation import Evaluation, mape
from meli_forecast.schemas import ForecastSchema, MetricsSchema, SplitSchema
from tests.utils import assert_pyspark_df_equal

group_columns = ["city", "product_id"]
time_column = "date"
target_column = "sales"
metrics = ["mae", "rmse", mape(epsilon=0)]
model_selection_metric = "mae"


def test_evaluation(spark: SparkSession):
    df_split = spark.createDataFrame(
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
    df_test = df_split.filter(df_split.split == "test")
    df_forecast_on_test = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 2 + ["Prophet"] * 2,
                "city": ["B1"] * 4,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 4,
                "date": [date(2024, 8, 6), date(2024, 8, 7)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )
    df_all_models_forecast = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 2 + ["Prophet"] * 2,
                "city": ["B1"] * 4,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 4,
                "date": [date(2024, 8, 8), date(2024, 8, 9)] * 2,
                "sales": [7.3, 11.8, 8.5, 10.7],
            }
        ),
        schema=ForecastSchema,
    )

    df_expected_metrics = spark.createDataFrame(
        pd.DataFrame(
            {
                "model": ["ExponentialSmoothing"] * 3 + ["Prophet"] * 3,
                "city": ["B1"] * 6,
                "product_id": ["8fa69e11-c148-470d-a4f1-1a1781079435"] * 6,
                "metric": ["rmse", "mae", "mape", "rmse", "mae", "mape"],
                "value": [1.750714, 1.75, 0.184444, 0.608276, 0.6, 0.062778],
            }
        ),
        schema=MetricsSchema,
    )
    df_expected_best_models = spark.createDataFrame(
        [
            Row(
                model="Prophet",
                city="B1",
                product_id="8fa69e11-c148-470d-a4f1-1a1781079435",
                metric="mae",
                value=0.600,
            )
        ],
        schema=MetricsSchema,
    )
    df_expected_forecast = spark.createDataFrame(
        [
            Row(
                model="Prophet",
                city="B1",
                product_id="8fa69e11-c148-470d-a4f1-1a1781079435",
                date=date(2024, 8, 8),
                sales=8.5,
            ),
            Row(
                model="Prophet",
                city="B1",
                product_id="8fa69e11-c148-470d-a4f1-1a1781079435",
                date=date(2024, 8, 9),
                sales=10.7,
            ),
        ],
        schema=ForecastSchema,
    )

    evaluation = Evaluation(
        group_columns,
        time_column,
        target_column,
        metrics,
        model_selection_metric,
    )
    df_metrics = evaluation.get_metrics(
        df_test, df_forecast_on_test, MetricsSchema
    )
    df_best_models = evaluation.get_best_models(df_metrics)
    df_forecast = evaluation.get_forecast(
        df_all_models_forecast, df_best_models
    )

    assert_pyspark_df_equal(df_expected_metrics, df_metrics, 2)
    assert_pyspark_df_equal(df_expected_best_models, df_best_models, 2)
    assert_pyspark_df_equal(df_expected_forecast, df_forecast)
