"""
build_row_key.py

Builds the single ROW_KEY column used to match rows between the SAS output and
the PySpark output, so DataFrameComparator can join on one column.

The key is the seven columns that identify a row uniquely in df_sas_asset:

    ENTITY_ID, ENTRY_LIFE_AGR, ASSET_START_DATE, ALM_SEGMENT, ASSETS_CODE_AGR,
    RISK_SII_VALUE, ACCOUNTING_VALUE_IFRS

Each value is normalised before it goes into the key, because the two sides
write the same value differently:

    text      trimmed, so 'AAA' and 'AAA  ' give the same key
    dates     formatted yyyy-MM-dd, so a date and a timestamp give the same key
    numbers   cast to a fixed number of decimals, so 1000, 1000.0 and
              999.9999999999999 give the same key, and -0.0 becomes 0.00
    null      replaced by '~', so rows with no ASSET_START_DATE still join
              (58% of the asset file has none, and null = null is false in Spark)
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T

KEY_COLUMNS = [
    "ENTITY_ID",
    "ENTRY_LIFE_AGR",
    "ASSET_START_DATE",
    "ALM_SEGMENT",
    "ASSETS_CODE_AGR",
    "RISK_SII_VALUE",
    "ACCOUNTING_VALUE_IFRS",
]

# decimals to keep per numeric key column; anything not listed uses DEFAULT_SCALE
NUMERIC_SCALE = {
    "ENTRY_LIFE_AGR": 0,          # a record id, no decimals
    "RISK_SII_VALUE": 2,
    "ACCOUNTING_VALUE_IFRS": 2,
}
DEFAULT_SCALE = 2
NULL_MARKER = "~"


def key_part(df, column):
    """Normalise one column into the string that goes into the key."""

    data_type = df.schema[column].dataType

    if isinstance(data_type, (T.DateType, T.TimestampType)):
        part = F.date_format(F.col(column), "yyyy-MM-dd")
    elif isinstance(data_type, (T.FloatType, T.DoubleType, T.DecimalType,
                                T.IntegerType, T.LongType, T.ShortType)):
        scale = NUMERIC_SCALE.get(column, DEFAULT_SCALE)
        part = F.col(column).cast(T.DecimalType(38, scale)).cast("string")
    else:
        part = F.trim(F.col(column).cast("string"))

    return F.coalesce(part, F.lit(NULL_MARKER))


def add_row_key(df, key_columns=None, out_column="ROW_KEY"):
    """Add the ROW_KEY column. Run this on both DataFrames before comparing."""

    key_columns = key_columns or KEY_COLUMNS

    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError("Key columns not found in the DataFrame: %s" % missing)

    parts = [key_part(df, c) for c in key_columns]
    return df.withColumn(out_column, F.concat_ws("|", *parts))


def check_row_key(df, df_name="dataframe", out_column="ROW_KEY", show=5):
    """Report whether ROW_KEY is unique. Run on both sides before comparing."""

    total = df.count()
    distinct = df.select(out_column).distinct().count()
    duplicates = (df.groupBy(out_column).count()
                    .filter(F.col("count") > 1)
                    .orderBy(F.col("count").desc()))
    dup_groups = duplicates.count()
    dup_rows = total - distinct

    print("%s: %d rows, %d distinct keys" % (df_name, total, distinct))
    if dup_groups:
        print("  NOT unique: %d keys are shared by more than one row (%d extra rows)"
              % (dup_groups, dup_rows))
        duplicates.show(show, truncate=False)
    else:
        print("  unique: ROW_KEY identifies every row")

    return {"rows": total, "distinct_keys": distinct,
            "duplicate_keys": dup_groups, "duplicate_rows": dup_rows}


# ---------------------------------------------------------------------------
# Example usage in a Synapse notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from dataframe_comparator import DataFrameComparator

    spark = SparkSession.builder.getOrCreate()

    sas_df = add_row_key(spark.table("dbo.df_sas_asset"))
    spark_df = add_row_key(spark.table("dbo.df_pyspark_asset"))

    check_row_key(sas_df, "SAS_asset")
    check_row_key(spark_df, "Spark_asset")

    report = DataFrameComparator(
        spark, sas_df, spark_df,
        key_columns=["ROW_KEY"],
        df1_name="SAS_asset",
        df2_name="Spark_asset",
        numeric_precision=2,
    ).compare()
