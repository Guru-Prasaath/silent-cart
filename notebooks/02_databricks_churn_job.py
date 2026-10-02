# Databricks notebook source
# MAGIC %md
# MAGIC # FreshBasket churn: Databricks production job (reference implementation)
# MAGIC
# MAGIC How the local pipeline maps onto an Azure + Databricks stack:
# MAGIC
# MAGIC | Step | Local project | Azure / Databricks |
# MAGIC |---|---|---|
# MAGIC | Ingest | `data/raw/*.xlsx` | Azure Data Factory copies the CRM / POS extracts into ADLS Gen2 (bronze) |
# MAGIC | Clean | `src/data_quality.py` | This notebook writes silver Delta tables (same rules) |
# MAGIC | Features | `src/features.py` | `src/spark_features.py` (parity-tested against pandas: `scripts/spark_parity_check.py`) |
# MAGIC | Train / track | `src/models.py` + local MLflow | Databricks MLflow + Unity Catalog model registry |
# MAGIC | Score | `outputs/churn_predictions.csv`, `src/api.py` | Monthly batch job writes a gold table, which feeds the CRM next-best-action queue |
# MAGIC
# MAGIC **Status:** the Spark feature code is executed and verified locally (Spark 4.2, Java 21). This notebook has not
# MAGIC been run on a Databricks workspace; table names are placeholders.

# COMMAND ----------

dbutils.widgets.text("cutoff", "2024-04-01")
dbutils.widgets.text("catalog", "freshbasket")
cutoff = dbutils.widgets.get("cutoff")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md ## 1. Silver tables (cleaned with the rules documented in outputs/data_quality_log.csv)

# COMMAND ----------

from pyspark.sql import functions as F

profile = spark.table(f"{catalog}.silver.customer_profile")
activity = spark.table(f"{catalog}.silver.monthly_activity")

# COMMAND ----------

# MAGIC %md ## 2. Leakage-safe feature snapshot (features use months before `cutoff` only)

# COMMAND ----------

from src.spark_features import build_snapshot_spark

snapshot = build_snapshot_spark(spark, profile, activity, cutoff)
(snapshot.withColumn("cutoff", F.lit(cutoff))
 .write.mode("overwrite").option("replaceWhere", f"cutoff = '{cutoff}'")
 .saveAsTable(f"{catalog}.gold.churn_features"))

# COMMAND ----------

# MAGIC %md ## 3. Score with the registered champion model and publish the action list

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")
model = mlflow.pyfunc.load_model(f"models:/{catalog}.ml.churn_model@champion")
scored = snapshot.toPandas()                       # ~2.6k members; use mlflow.pyfunc.spark_udf at millions
scored["churn_probability"] = model.predict(scored)
spark.createDataFrame(scored).write.mode("overwrite").saveAsTable(f"{catalog}.gold.churn_scores")
