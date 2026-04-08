from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, when, explode, array, struct, coalesce
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()

# Path to Excel in Azure (ADLS / Blob)
file_path = "abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx"

# Sheet name (this becomes report_id)
sheet_name = "sheet3"  # change as needed

# ====================================================
# READ EXCEL FILE
# ====================================================
df = spark.read.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("dataAddress", f"'{sheet_name}'!A1") \
    .load(file_path)

print("Original DataFrame:")
df.show(10, truncate=False)
print(f"Columns: {df.columns}")

# ====================================================
# STEP 1: Get the first column name (will be row_id)
# ====================================================
first_col = df.columns[0]
print(f"First column (row_id): {first_col}")

# Rename first column to row_id
df = df.withColumnRenamed(first_col, "row_id")

# ====================================================
# STEP 2: Add report_id column (worksheet name)
# ====================================================
df = df.withColumn("report_id", lit(sheet_name))

# ====================================================
# STEP 3: Filter out header rows and empty rows
# ====================================================
# Remove rows where row_id is null or empty or contains descriptive text
df = df.filter(
    (col("row_id").isNotNull()) & 
    (col("row_id") != "") & 
    (col("row_id").like("R%"))  # Keep only rows starting with R (R0010, R0020, etc.)
)

print("After filtering:")
df.show(10, truncate=False)

# ====================================================
# STEP 4: Cast all value columns to String and replace NULLs with empty space
# ====================================================
value_columns = [c for c in df.columns if c not in ["report_id", "row_id"]]

for c in value_columns:
    df = df.withColumn(
        c,
        coalesce(col(c).cast(StringType()), lit(" "))
    )

print("After null replacement:")
df.show(10, truncate=False)

# ====================================================
# STEP 5: Unpivot using explode + array of structs
# ====================================================
# Create array of structs: each struct contains column_id (C0020, C0030, etc.) and its value
struct_cols = [
    struct(lit(c).alias("column_id"), col(c).alias("value")) 
    for c in value_columns
]

# Select report_id, row_id, and explode the array of column structs
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

# ====================================================
# STEP 6: Filter out rows where value is empty
# ====================================================
# Remove rows where value is empty, null, or just whitespace
df_unpivot = df_unpivot.filter(
    (col("value").isNotNull()) & 
    (col("value").trim() != "") &
    (col("value") != " ")
)

print("Final unpivoted data (blanks removed):")
df_unpivot.show(50, truncate=False)

# ====================================================
# STEP 7: Write to Azure
# ====================================================
output_path = "abfss://<container>@<storage>.dfs.core.windows.net/<output_path>/unpivoted_data"

# Parquet format (recommended for performance)
df_unpivot.write.mode("overwrite").parquet(output_path)
print(f"✓ Data written to Parquet: {output_path}")

# Also save as CSV if needed for Excel export
output_csv = "abfss://<container>@<storage>.dfs.core.windows.net/<output_path>/unpivoted_data.csv"
df_unpivot.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_csv)
print(f"✓ Data written to CSV: {output_csv}")

# ====================================================
# OPTIONAL: Preview what will be exported to Excel
# ====================================================
print("\nFinal output (first 30 rows):")
df_unpivot.select("report_id", "row_id", "column_id", "value").show(30, truncate=False)

# Count total rows
print(f"Total rows in unpivoted data: {df_unpivot.count()}")
