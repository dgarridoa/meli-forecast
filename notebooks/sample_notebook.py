# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Sample notebook

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Aux steps for auto reloading of dependent files

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Example usage of existing code

# COMMAND ----------

from pathlib import Path

import yaml
from pyspark.sql import SparkSession

from meli_forecast.params import Params
from meli_forecast.tasks.split import SplitTask

# COMMAND ----------

project_root = Path(".").absolute().parent

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
spark.sql("USE CATALOG `meli-forecast`")

# COMMAND ----------

conf_file = f"{project_root}/conf/dev_config.yml"
conf = yaml.safe_load(Path(project_root, conf_file).read_text())
params = Params(**conf)

# COMMAND ----------

task = SplitTask(params.split)
task.launch(spark)
