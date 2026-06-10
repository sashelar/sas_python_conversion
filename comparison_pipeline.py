# =========================================================================
# SYNAPSE NOTEBOOK CELL - Comparison Pipeline
# Scans two ADLS folders, matches datasets by name, runs comparator in loop,
# writes flat summary + detail parquet back to ADLS.
# =========================================================================

"""
Folder convention:
  SAS files    → SAS_BASE_PATH/  (e.g. dataset_name.sas7bdat)
  PySpark data → SPARK_BASE_PATH/ (e.g. dataset_name.parquet or dataset_name/ folder)

Results written to:
  RESULTS_BASE_PATH/summary/   ← one row per dataset, latest run wins
  RESULTS_BASE_PATH/detail/    ← one row per column per dataset
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType,
    IntegerType, DoubleType, LongType, TimestampType
)
from datetime import datetime
from typing import List, Optional, Tuple
import json
import uuid

# ── Paste your DataFrameComparator class above this cell or import it ──────
# from dataframe_comparator import DataFrameComparator

# =========================================================================
# CONFIGURATION  ← edit these for your environment
# =========================================================================

SAS_BASE_PATH   = "abfss://<container>@<storage>.dfs.core.windows.net/sas/"
SPARK_BASE_PATH = "abfss://<container>@<storage>.dfs.core.windows.net/pyspark/"
RESULTS_BASE_PATH = "abfss://<container>@<storage>.dfs.core.windows.net/comparison_results/"

# Columns used as join keys per dataset — None means row-order comparison
# Format: {"dataset_name": ["col1", "col2"], ...}  — omit a name to use None
KEY_COLUMNS_MAP = {
    # "claims":   ["claim_id"],
    # "members":  ["member_id", "effective_date"],
}

NUMERIC_PRECISION = 6
FUZZY_THRESHOLD   = 0.8

# =========================================================================
# SAS READER HOOK  ← plug in your existing function here
# =========================================================================

def load_sas_dataset(sas_path: str, spark: SparkSession) -> DataFrame:
    """
    Replace the body of this function with your existing SAS reader logic.
    Must return a Spark DataFrame.
    """
    # ---- your code here ----
    # Example using pandas + pyreadstat:
    # import pandas as pd, pyreadstat
    # pdf, meta = pyreadstat.read_sas7bdat(sas_path)
    # return spark.createDataFrame(pdf)
    raise NotImplementedError(f"Plug in your SAS reader for: {sas_path}")


# =========================================================================
# HELPER: list files in an ADLS folder via Spark Hadoop FS API
# =========================================================================

def list_adls_files(path: str, spark: SparkSession) -> List[str]:
    """Return list of file/folder names (not full paths) under an ADLS path."""
    jvm  = spark._jvm
    conf = spark._jsc.hadoopConfiguration()
    fs   = jvm.org.apache.hadoop.fs.FileSystem.get(
               jvm.java.net.URI.create(path), conf)
    statuses = fs.listStatus(jvm.org.apache.hadoop.fs.Path(path))
    return [str(s.getPath().getName()) for s in statuses]


def strip_extension(name: str) -> str:
    """Remove known data-file extensions to get the base dataset name."""
    for ext in (".sas7bdat", ".parquet", ".csv", ".json"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


# =========================================================================
# HELPER: flatten a comparator report into summary + detail rows
# =========================================================================

def flatten_report(
    report: dict,
    dataset_name: str,
    run_id: str,
    run_ts: str,
    df1_path: str,
    df2_path: str,
) -> Tuple[dict, List[dict]]:
    """
    Returns:
        summary_row  — one dict for the summary parquet
        detail_rows  — list of dicts, one per column, for the detail parquet
    """
    rc = report.get("row_count", {})
    cols_rpt = report.get("columns", {})
    schema_rpt = report.get("schema", {})
    data_rpt = report.get("data_comparison", {})

    col_details = data_rpt.get("column_details", {})
    total_cols    = len(col_details) if col_details else cols_rpt.get("column_count_df1", 0)
    matching_cols = sum(1 for d in col_details.values() if d.get("all_values_match", False))
    match_pct     = round(matching_cols / total_cols * 100, 2) if total_cols > 0 else 0.0

    summary_row = {
        "run_id":           run_id,
        "dataset_name":     dataset_name,
        "run_timestamp":    run_ts,
        "overall_match":    report.get("overall_match", False),
        "row_count_match":  rc.get("match", False),
        "columns_match":    report.get("columns_match", False),
        "schema_match":     schema_rpt.get("schemas_match", False),
        "data_status":      data_rpt.get("status", "SKIPPED"),
        "df1_row_count":    rc.get("df1_count", 0),
        "df2_row_count":    rc.get("df2_count", 0),
        "row_diff":         rc.get("difference", 0),
        "total_columns":    total_cols,
        "matching_columns": matching_cols,
        "match_pct":        match_pct,
        "df1_path":         df1_path,
        "df2_path":         df2_path,
        "error":            None,
    }

    detail_rows = []
    for col_name, d in col_details.items():
        stats = d.get("statistics", {})
        detail_rows.append({
            "run_id":             run_id,
            "dataset_name":       dataset_name,
            "run_timestamp":      run_ts,
            "column_name":        col_name,
            "data_type":          d.get("data_type", "unknown"),
            "comparison_type":    d.get("comparison_type", "unknown"),
            "all_values_match":   d.get("all_values_match", False),
            "match_pct":          d.get("match_percentage", 0.0),
            "exact_matches":      d.get("exact_matches", 0),
            "mismatches":         d.get("mismatches", 0),
            "avg_diff":           stats.get("average_difference"),
            "min_diff":           stats.get("min_difference"),
            "max_diff":           stats.get("max_difference"),
            "avg_fuzzy_score":    stats.get("average_fuzzy_score"),
            "min_fuzzy_score":    stats.get("min_fuzzy_score"),
            "sample_mismatches":  json.dumps(d.get("sample_mismatches", []), default=str),
        })

    return summary_row, detail_rows


# =========================================================================
# PARQUET SCHEMAS
# =========================================================================

SUMMARY_SCHEMA = StructType([
    StructField("run_id",           StringType(),  True),
    StructField("dataset_name",     StringType(),  True),
    StructField("run_timestamp",    StringType(),  True),
    StructField("overall_match",    BooleanType(), True),
    StructField("row_count_match",  BooleanType(), True),
    StructField("columns_match",    BooleanType(), True),
    StructField("schema_match",     BooleanType(), True),
    StructField("data_status",      StringType(),  True),
    StructField("df1_row_count",    LongType(),    True),
    StructField("df2_row_count",    LongType(),    True),
    StructField("row_diff",         LongType(),    True),
    StructField("total_columns",    IntegerType(), True),
    StructField("matching_columns", IntegerType(), True),
    StructField("match_pct",        DoubleType(),  True),
    StructField("df1_path",         StringType(),  True),
    StructField("df2_path",         StringType(),  True),
    StructField("error",            StringType(),  True),
])

DETAIL_SCHEMA = StructType([
    StructField("run_id",           StringType(),  True),
    StructField("dataset_name",     StringType(),  True),
    StructField("run_timestamp",    StringType(),  True),
    StructField("column_name",      StringType(),  True),
    StructField("data_type",        StringType(),  True),
    StructField("comparison_type",  StringType(),  True),
    StructField("all_values_match", BooleanType(), True),
    StructField("match_pct",        DoubleType(),  True),
    StructField("exact_matches",    LongType(),    True),
    StructField("mismatches",       LongType(),    True),
    StructField("avg_diff",         DoubleType(),  True),
    StructField("min_diff",         DoubleType(),  True),
    StructField("max_diff",         DoubleType(),  True),
    StructField("avg_fuzzy_score",  DoubleType(),  True),
    StructField("min_fuzzy_score",  DoubleType(),  True),
    StructField("sample_mismatches", StringType(), True),
])


# =========================================================================
# MAIN PIPELINE
# =========================================================================

def run_comparison_pipeline(spark: SparkSession):
    run_id = str(uuid.uuid4())[:8]
    run_ts = datetime.now().isoformat()

    print(f"Run ID     : {run_id}")
    print(f"Timestamp  : {run_ts}")
    print(f"SAS path   : {SAS_BASE_PATH}")
    print(f"Spark path : {SPARK_BASE_PATH}")
    print("=" * 70)

    # Discover datasets in both folders
    sas_files   = list_adls_files(SAS_BASE_PATH,   spark)
    spark_files = list_adls_files(SPARK_BASE_PATH, spark)

    sas_map   = {strip_extension(f): f for f in sas_files}
    spark_map = {strip_extension(f): f for f in spark_files}

    matched   = sorted(set(sas_map) & set(spark_map))
    only_sas  = sorted(set(sas_map) - set(spark_map))
    only_spk  = sorted(set(spark_map) - set(sas_map))

    print(f"Matched datasets   : {len(matched)}")
    if only_sas:
        print(f"Only in SAS folder : {only_sas}")
    if only_spk:
        print(f"Only in Spark folder: {only_spk}")
    print()

    all_summary_rows = []
    all_detail_rows  = []

    for dataset_name in matched:
        sas_path   = SAS_BASE_PATH   + sas_map[dataset_name]
        spark_path = SPARK_BASE_PATH + spark_map[dataset_name]

        print(f"[{dataset_name}] Comparing...")

        try:
            df1 = load_sas_dataset(sas_path, spark)
            df2 = spark.read.parquet(spark_path)

            key_cols = KEY_COLUMNS_MAP.get(dataset_name)

            comparator = DataFrameComparator(
                spark=spark,
                df1=df1,
                df2=df2,
                key_columns=key_cols,
                numeric_precision=NUMERIC_PRECISION,
                fuzzy_threshold=FUZZY_THRESHOLD,
            )
            report = comparator.compare()

            summary_row, detail_rows = flatten_report(
                report, dataset_name, run_id, run_ts, sas_path, spark_path
            )
            status = "PASS" if summary_row["overall_match"] else "FAIL"
            print(f"[{dataset_name}] {status} — {summary_row['match_pct']}% data match\n")

        except Exception as e:
            print(f"[{dataset_name}] ERROR: {e}\n")
            summary_row = {
                "run_id": run_id, "dataset_name": dataset_name,
                "run_timestamp": run_ts, "overall_match": False,
                "row_count_match": False, "columns_match": False,
                "schema_match": False, "data_status": "ERROR",
                "df1_row_count": None, "df2_row_count": None,
                "row_diff": None, "total_columns": None,
                "matching_columns": None, "match_pct": None,
                "df1_path": sas_path, "df2_path": spark_path,
                "error": str(e),
            }
            detail_rows = []

        all_summary_rows.append(summary_row)
        all_detail_rows.extend(detail_rows)

    # Write results to ADLS parquet (overwrite latest run per dataset via merge key)
    _write_results(spark, all_summary_rows, all_detail_rows)

    print("=" * 70)
    print(f"Pipeline complete. {len(matched)} datasets processed.")
    print(f"Results written to: {RESULTS_BASE_PATH}")

    return all_summary_rows


def _write_results(spark, summary_rows, detail_rows):
    summary_path = RESULTS_BASE_PATH + "summary/"
    detail_path  = RESULTS_BASE_PATH + "detail/"

    if summary_rows:
        summary_df = spark.createDataFrame(summary_rows, schema=SUMMARY_SCHEMA)
        (summary_df
            .write
            .mode("overwrite")
            .partitionBy("dataset_name")
            .parquet(summary_path))
        print(f"Summary written : {summary_path}")

    if detail_rows:
        detail_df = spark.createDataFrame(detail_rows, schema=DETAIL_SCHEMA)
        (detail_df
            .write
            .mode("overwrite")
            .partitionBy("dataset_name")
            .parquet(detail_path))
        print(f"Detail written  : {detail_path}")


# =========================================================================
# RUN
# =========================================================================
# run_comparison_pipeline(spark)
