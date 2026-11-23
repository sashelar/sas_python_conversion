"""
clean_for_compare.py

Normalises the SAS output and the PySpark output so DataFrameComparator only
reports real differences, not the two sides writing the same value differently.

Found by comparing df_sas_asset against df_py_enriched_testing (36,362 rows,
255 columns, joined on ROW_KEY):

    column name with spaces   PySpark has "SCR APPROACH FOR CIU"
    trailing spaces           CIC_POS_1_DESC: 'Investment funds ' in PySpark, 2,996 rows
    upper-cased values        SII_SOURCE: 'SII_Value_DB' in SAS, 'SII_VALUE_DB' in
                              PySpark, 34,128 rows
    zero written as null      COUNTRY_LMT (4,270), SOVEREIGN_LMT (3,996),
                              LIMITS_ISSUER (106): 0 in SAS, null in PySpark
    float noise               27 numeric columns differ by up to 2e-8 absolute
                              (4e-8 relative) - handled by numeric_precision in
                              the comparator, nothing to clean

Everything that survives this cleaning is a real difference worth investigating.
"""

from pyspark.sql import functions as F
from pyspark.sql import types as T

# columns where SAS writes 0 and PySpark writes null
FILL_ZERO_COLUMNS = ["COUNTRY_LMT", "SOVEREIGN_LMT", "LIMITS_ISSUER"]

# columns where the two sides differ only in capitalisation. The real fix is to
# drop the upper() in the PySpark pipeline; until then, compare case-insensitively.
UPPER_CASE_COLUMNS = ["SII_SOURCE"]


def clean_column_names(df):
    """Upper-case names and replace spaces with underscores."""
    for name in df.columns:
        fixed = name.strip().upper().replace(" ", "_")
        if fixed != name:
            df = df.withColumnRenamed(name, fixed)
    return df


def clean_values(df, fill_zero_columns=None, upper_case_columns=None):
    """Trim text, turn blanks into null, and apply the two fixes listed above."""

    fill_zero_columns = FILL_ZERO_COLUMNS if fill_zero_columns is None else fill_zero_columns
    upper_case_columns = UPPER_CASE_COLUMNS if upper_case_columns is None else upper_case_columns

    for field in df.schema.fields:
        name = field.name

        if isinstance(field.dataType, T.StringType):
            trimmed = F.trim(F.col(name))
            if name in upper_case_columns:
                trimmed = F.upper(trimmed)
            df = df.withColumn(name, F.when(trimmed == "", None).otherwise(trimmed))

        elif name in fill_zero_columns:
            df = df.withColumn(name, F.coalesce(F.col(name), F.lit(0).cast(field.dataType)))

    return df


def clean(df, fill_zero_columns=None, upper_case_columns=None):
    """Both steps. Run on the SAS side and the PySpark side."""
    return clean_values(clean_column_names(df), fill_zero_columns, upper_case_columns)


def align_columns(df1, df2, df1_name="df1", df2_name="df2"):
    """Report columns that exist on one side only, and return the common ones."""

    only_1 = [c for c in df1.columns if c not in df2.columns]
    only_2 = [c for c in df2.columns if c not in df1.columns]
    common = [c for c in df1.columns if c in df2.columns]

    if only_1:
        print("only in %s (%d): %s" % (df1_name, len(only_1), only_1))
    if only_2:
        print("only in %s (%d): %s" % (df2_name, len(only_2), only_2))
    print("comparing %d common columns" % len(common))

    return df1.select(*common), df2.select(*common)


# ---------------------------------------------------------------------------
# Example usage in a Synapse notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from build_row_key import add_row_key, check_row_key
    from dataframe_comparator import DataFrameComparator

    spark = SparkSession.builder.getOrCreate()

    sas_df = clean(spark.table("dbo.df_sas_asset"))
    spark_df = clean(spark.table("dbo.df_py_enriched"))

    sas_df, spark_df = align_columns(sas_df, spark_df, "SAS_asset", "Spark_asset")

    sas_df = add_row_key(sas_df)
    spark_df = add_row_key(spark_df)
    check_row_key(sas_df, "SAS_asset")
    check_row_key(spark_df, "Spark_asset")

    report = DataFrameComparator(
        spark, sas_df, spark_df,
        key_columns=["ROW_KEY"],
        df1_name="SAS_asset",
        df2_name="Spark_asset",
        numeric_precision=2,          # absorbs the float noise
    ).compare()
