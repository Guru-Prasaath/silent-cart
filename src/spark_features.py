"""PySpark port of the snapshot feature builder, for running the pipeline on Databricks at scale.

Same logic as features.build_snapshot (verified by scripts/spark_parity_check.py):
features use months strictly before `cutoff`; inside the 6-month window, trailing silence after a member's
last row counts as 0, gaps between rows and months before the first row count as missing.
On Databricks, `activity` / `profile` would be Delta tables landed by Azure Data Factory.
"""
from pyspark.sql import DataFrame, SparkSession, functions as F

W = 6
CAP = W + 1
METRICS = ["TRANSACTIONS", "TOTAL_SPEND", "APP_SESSIONS", "EMAILS_OPENED", "COUPONS_REDEEMED", "SUPPORT_TICKETS", "COMPLAINT_FLAG"]
MEAN_METRICS = {"TRANSACTIONS": "txn", "TOTAL_SPEND": "spend", "APP_SESSIONS": "app", "EMAILS_OPENED": "email",
                "COUPONS_REDEEMED": "coupons"}   # monthly means; support metrics are sums


def _mi(col):
    return F.year(col) * 12 + F.month(col)


def _pct_change(recent, prior):
    v = (recent - prior) / F.greatest(prior, F.lit(0.5))
    return F.least(F.greatest(v, F.lit(-1.0)), F.lit(3.0))


def build_snapshot_spark(spark: SparkSession, profile: DataFrame, activity: DataFrame, cutoff: str,
                         label_months: int = 3) -> DataFrame:
    c = spark.range(1).select(_mi(F.to_date(F.lit(cutoff))).alias("c")).first()["c"]
    act = activity.withColumn("k", F.lit(c) - _mi(F.col("MONTH")))
    hist = act.where("k >= 1")

    stats = hist.groupBy("CUSTOMER_ID").agg(
        F.max("k").alias("first_k"), F.min("k").alias("last_k"),
        F.min(F.when(F.col("TRANSACTIONS") > 0, F.col("k"))).alias("last_buy_k"),
        F.min(F.when(F.col("SUPPORT_TICKETS") > 0, F.col("k"))).alias("last_ticket_k"))
    eligible = (stats.where("last_buy_k IS NOT NULL")
                .join(profile.where(F.col("SIGNUP_DATE") < F.to_date(F.lit(cutoff))), "CUSTOMER_ID"))

    # Dense customer x month grid over the 6-month lookback, then apply the missing-month rules.
    grid = eligible.select("CUSTOMER_ID", "first_k", "last_k").crossJoin(
        spark.range(1, W + 1).select(F.col("id").cast("int").alias("k")))
    cols = [F.col(m) for m in METRICS]
    g = grid.join(hist.where(f"k <= {W}").select("CUSTOMER_ID", "k", *cols), ["CUSTOMER_ID", "k"], "left")
    # Materialise the row status BEFORE filling, so every metric sees the original presence flag.
    g = g.withColumn("present", F.col("TRANSACTIONS").isNotNull())
    g = g.withColumn("trailing", (F.col("k") < F.col("last_k")) & ~F.col("present"))
    g = g.withColumn("is_gap", (~F.col("present") & ~F.col("trailing") & (F.col("k") <= F.col("first_k"))).cast("int"))
    for m in METRICS:
        g = g.withColumn(m, F.when(F.col("trailing"), F.lit(0.0)).otherwise(F.col(m).cast("double")))
    g = g.withColumn("x", F.when(F.col("TRANSACTIONS").isNotNull(), -F.col("k").cast("double")))

    aggs = [F.sum("is_gap").alias("missing_months_6m")]
    for m, s in MEAN_METRICS.items():
        aggs += [F.avg(F.when(F.col("k") <= 3, F.col(m))).alias(f"{s}_l3_raw"),
                 F.avg(F.when(F.col("k") > 3, F.col(m))).alias(f"{s}_p3_raw")]
    aggs += [
        F.sum(F.when(F.col("k") <= 3, F.coalesce("SUPPORT_TICKETS", F.lit(0.0)))).alias("tickets_l3"),
        F.sum(F.when(F.col("k") > 3, F.coalesce("SUPPORT_TICKETS", F.lit(0.0)))).alias("tickets_p3"),
        F.sum(F.coalesce("COMPLAINT_FLAG", F.lit(0.0))).alias("complaints_6m"),
        F.max(F.when(F.col("k") <= 3, F.coalesce("COMPLAINT_FLAG", F.lit(0.0)))).alias("complaint_l3"),
        F.sum(F.when((F.col("k") <= 3) & (F.col("TRANSACTIONS") > 0), 1).otherwise(0)).alias("active_months_l3"),
        F.count("TRANSACTIONS").alias("n_obs"),
        F.sum(F.when(F.col("TRANSACTIONS") > 0, 1).otherwise(0)).alias("n_active"),
        F.count("x").alias("n_x"),
        F.try_divide(F.covar_pop("x", "TRANSACTIONS"), F.var_pop("x")).alias("txn_slope_raw"),
        F.try_divide(F.covar_pop(F.when(F.col("APP_SESSIONS").isNotNull(), F.col("x")), "APP_SESSIONS"),
                     F.var_pop(F.when(F.col("APP_SESSIONS").isNotNull(), F.col("x")))).alias("app_slope_raw"),
        F.stddev_pop("TOTAL_SPEND").alias("spend_sd"), F.avg("TOTAL_SPEND").alias("spend_mean"),
    ]
    w = g.groupBy("CUSTOMER_ID").agg(*aggs)

    out = eligible.join(w, "CUSTOMER_ID")
    for s in MEAN_METRICS.values():
        l3, p3 = F.col(f"{s}_l3_raw"), F.col(f"{s}_p3_raw")
        out = (out.withColumn(f"{s}_l3", F.coalesce(l3, p3, F.lit(0.0)))
                  .withColumn(f"{s}_p3", F.coalesce(p3, l3, F.lit(0.0))))
    fut = (act.where(f"k <= 0 AND k > {-label_months}").groupBy("CUSTOMER_ID")
           .agg(F.sum("TRANSACTIONS").alias("future_txn")))
    out = out.join(fut, "CUSTOMER_ID", "left")
    return out.select(
        F.col("CUSTOMER_ID").alias("customer_id"),
        F.least(F.col("last_buy_k"), F.lit(CAP)).alias("recency_months"),
        F.col("txn_l3"), F.col("txn_p3"), _pct_change(F.col("txn_l3"), F.col("txn_p3")).alias("txn_change_pct"),
        F.col("spend_l3"), F.col("spend_p3"), _pct_change(F.col("spend_l3"), F.col("spend_p3")).alias("spend_change_pct"),
        F.when(F.col("n_x") >= 3, F.coalesce("txn_slope_raw", F.lit(0.0))).otherwise(0.0).alias("txn_slope_6m"),
        F.coalesce(F.col("spend_sd") / F.greatest(F.col("spend_mean"), F.lit(1.0)), F.lit(0.0)).alias("spend_cv_6m"),
        F.col("active_months_l3"),
        (F.col("n_active") / F.greatest(F.col("n_obs"), F.lit(1))).alias("active_ratio_6m"),
        F.col("app_l3"), F.col("app_p3"), _pct_change(F.col("app_l3"), F.col("app_p3")).alias("app_change_pct"),
        F.when(F.col("n_x") >= 3, F.coalesce("app_slope_raw", F.lit(0.0))).otherwise(0.0).alias("app_slope_6m"),
        F.col("email_l3"), F.col("email_p3"), F.col("coupons_l3"), F.col("coupons_p3"),
        F.col("tickets_l3"), F.col("tickets_p3"), (F.col("tickets_l3") + F.col("tickets_p3")).alias("tickets_6m"),
        F.col("complaints_6m"), F.col("complaint_l3").cast("int").alias("complaint_l3"),
        F.least(F.coalesce("last_ticket_k", F.lit(CAP)), F.lit(CAP)).alias("months_since_ticket"),
        F.least(F.col("first_k"), F.lit(W)).alias("history_months"),
        (F.col("first_k") < 4).cast("int").alias("short_history"),
        F.col("missing_months_6m"),
        (F.lit(c) - _mi(F.col("SIGNUP_DATE"))).alias("tenure_months"),
        F.when(F.col("last_buy_k") <= 3, "Active").otherwise("Lapsed").alias("segment"),
        (F.coalesce(F.col("future_txn"), F.lit(0)) == 0).cast("int").alias("churned"),
    )
