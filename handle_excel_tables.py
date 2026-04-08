from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when, explode, array, struct
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()

# Path to Excel in Azure (ADLS / Blob)
file_path = "abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx"

# Sheet name (this becomes report_id)
sheet_name = "Sheet1"  # change as needed

# ====================================================
# READ EXCEL FILE
# ====================================================
df = spark.read.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("dataAddress", f"'{sheet_name}'!A1") \
    .load(file_path)

print("Original DataFrame:")
df.show(5, truncate=False)

# ====================================================
# STEP 1: Rename first column as row_id
# ====================================================
first_col = df.columns[0]
df = df.withColumnRenamed(first_col, "row_id")

# ====================================================
# STEP 2: Add report_id column (worksheet name)
# ====================================================
df = df.withColumn("report_id", lit(sheet_name))

# ====================================================
# STEP 3: Replace NULLs with empty space
# ====================================================
for c in df.columns:
    df = df.withColumn(
        c,
        when(col(c).isNull(), lit(" "))
        .otherwise(col(c).cast(StringType()))
    )

print("After adding report_id and null handling:")
df.show(5, truncate=False)

# ====================================================
# STEP 4: Unpivot using explode + array of structs
# ====================================================
# Get all value columns (exclude report_id and row_id)
value_columns = [c for c in df.columns if c not in ["report_id", "row_id"]]

# Create array of structs for each column
struct_cols = [struct(lit(c).alias("column_id"), col(c).alias("value")) for c in value_columns]

# Explode the array
df_unpivot = df.select(
    "report_id",
    "row_id",
    explode(array(*struct_cols)).alias("data")
).select(
    "report_id",
    "row_id",
    col("data.column_id"),
    col("data.value")
)

print("After unpivoting:")
df_unpivot.show(20, truncate=False)

# ====================================================
# OPTIONAL: Write to Azure (Parquet, Delta, or CSV)
# ====================================================
# Parquet (recommended)
output_path = "abfss://<container>@<storage>.dfs.core.windows.net/<output_path>/unpivoted_data"
df_unpivot.write.mode("overwrite").parquet(output_path)

# Or Delta Lake:
# df_unpivot.write.mode("overwrite").format("delta").save(output_path)

# Or CSV:
# df_unpivot.write.mode("overwrite").option("header", "true").csv(output_path)

print(f"Data written to {output_path}")
