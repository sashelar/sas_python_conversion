from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, explode, array, struct, coalesce
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

print("=" * 80)
print("STEP 1: ORIGINAL DATA FROM EXCEL")
print("=" * 80)
print(f"Total rows: {df.count()}")
print(f"Columns: {df.columns}")
df.show(20, truncate=False)

# ====================================================
# STEP 2: Rename first column as row_id
# ====================================================
first_col = df.columns[0]
print(f"\nFirst column name: '{first_col}' -> will be renamed to 'row_id'")
df = df.withColumnRenamed(first_col, "row_id")

# ====================================================
# STEP 3: Add report_id column
# ====================================================
df = df.withColumn("report_id", lit(sheet_name))

print("\n" + "=" * 80)
print("STEP 2: AFTER ADDING report_id")
print("=" * 80)
print(f"Total rows: {df.count()}")
print(f"Columns: {df.columns}")
df.show(20, truncate=False)

# ====================================================
# STEP 4: Filter out header/empty rows (optional - remove if needed)
# ====================================================
df_before_filter = df.count()

# Remove rows where row_id is completely empty
df = df.filter(col("row_id").isNotNull() & (col("row_id") != ""))

df_after_filter = df.count()

print("\n" + "=" * 80)
print(f"STEP 3: FILTER ROWS (removed null/empty row_ids)")
print("=" * 80)
print(f"Rows before filter: {df_before_filter}")
print(f"Rows after filter: {df_after_filter}")
df.show(20, truncate=False)

# ====================================================
# STEP 5: Get value columns
# ====================================================
value_columns = [c for c in df.columns if c not in ["report_id", "row_id"]]
print(f"\nValue columns to unpivot: {value_columns}")

# ====================================================
# STEP 6: Cast to String and handle nulls
# ====================================================
for c in value_columns:
    df = df.withColumn(
        c,
        coalesce(col(c).cast(StringType()), lit(" "))
    )

print("\n" + "=" * 80)
print("STEP 4: AFTER CASTING TO STRING")
print("=" * 80)
print(f"Total rows: {df.count()}")
df.show(20, truncate=False)

# ====================================================
# STEP 7: UNPIVOT - THE CRITICAL PART
# ====================================================
print("\n" + "=" * 80)
print("STEP 5: UNPIVOTING...")
print("=" * 80)

# Create struct for each column
struct_cols = [
    struct(lit(c).alias("column_id"), col(c).alias("value")) 
    for c in value_columns
]

print(f"Creating array with {len(struct_cols)} columns...")

# Explode
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

print(f"Total rows after unpivot: {df_unpivot.count()}")

print("\n" + "=" * 80)
print("FINAL RESULT")
print("=" * 80)
df_unpivot.show(100, truncate=False)

# Count
total = df_unpivot.count()
print(f"\n✓ Total unpivoted rows: {total}")

# ====================================================
# Optional: Write to Azure
# ====================================================
output_path = "abfss://<container>@<storage>.dfs.core.windows.net/<output_path>/unpivoted_data"
output_csv = "abfss://<container>@<storage>.dfs.core.windows.net/<output_path>/unpivoted_data.csv"

try:
    df_unpivot.write.mode("overwrite").parquet(output_path)
    print(f"✓ Parquet written: {output_path}")
except Exception as e:
    print(f"✗ Parquet write failed: {e}")

try:
    df_unpivot.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_csv)
    print(f"✓ CSV written: {output_csv}")
except Exception as e:
    print(f"✗ CSV write failed: {e}")
