"""Verify the PySpark feature builder reproduces the pandas features exactly.

Requires Java 17 or 21 (set JAVA_HOME) and `pip install -r requirements-spark.txt`.
    python scripts/spark_parity_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from src import config as C, data_quality, features
from src.spark_features import METRICS, build_snapshot_spark


warnings.filterwarnings("ignore", category=FutureWarning)   # pyspark 4.2 notes pandas 3 support is partial
warnings.filterwarnings("ignore", category=UserWarning)     # Arrow fallback notice (pyarrow optional)


def main():
    profile, activity, _, _ = data_quality.run(save=False)
    spark = (SparkSession.builder.master("local[2]").appName("freshbasket-parity")
             .config("spark.ui.enabled", "false").config("spark.sql.session.timeZone", "UTC").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    # Pass calendar dates (not naive timestamps) so the JVM cannot shift them by the local timezone.
    prof_s = spark.createDataFrame(profile[["CUSTOMER_ID", "SIGNUP_DATE"]].assign(SIGNUP_DATE=profile.SIGNUP_DATE.dt.date))
    act_s = spark.createDataFrame(activity[["CUSTOMER_ID", "MONTH"] + list(METRICS)].assign(MONTH=activity.MONTH.dt.date))
    results = []
    for cutoff in C.TRAIN_CUTOFFS + [C.VALID_CUTOFF, C.TEST_CUTOFF]:
        pdf = features.build_snapshot(profile, activity, cutoff).set_index("customer_id").sort_index()
        sdf = build_snapshot_spark(spark, prof_s, act_s, cutoff.strftime("%Y-%m-%d")).toPandas().set_index("customer_id").sort_index()
        assert pdf.index.equals(sdf.index), f"eligible members differ at {cutoff.date()}"
        cols = [c for c in sdf.columns if c in pdf.columns]
        bad = [c for c in cols if not (np.allclose(pdf[c].astype(float), sdf[c].astype(float), atol=1e-6)
                                      if c != "segment" else (pdf[c] == sdf[c]).all())]
        results.append({"cutoff": cutoff.date(), "members": len(pdf), "features_compared": len(cols), "mismatched": bad})
        print(f"{cutoff.date()}: {len(pdf):,} members, {len(cols)} columns compared, mismatches: {bad or 'none'}")
    spark.stop()
    out = pd.DataFrame(results)
    out.to_csv(C.OUTPUTS / "spark_parity_check.csv", index=False)
    if any(r["mismatched"] for r in results):
        sys.exit(1)
    print("PySpark and pandas feature pipelines agree.")


if __name__ == "__main__":
    main()
