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
# STEP 2: Extract row_id (R0010, R0020, etc.) from 2nd column
# ====================================================
from pyspark.sql.functions import regexp_extract, when

# The row IDs are in the 2nd column (R0010, R0020, etc.)
second_col = df.columns[1]

print(f"Extracting row_id from 2nd column: '{second_col}'")

# Create row_id from 2nd column (clean up any extra text)
df = df.withColumn(
    "row_id",
    when(col(second_col).isNotNull(),
         regexp_extract(col(second_col), r'(R\d+)', 1)
    ).otherwise(col(second_col))
)

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
# STEP 5: Get value columns and map to proper column_ids
# ====================================================
# Exclude first 2 columns (first_col, second_col) and our new columns
first_col = df.columns[0]
second_col = df.columns[1]
value_columns = [c for c in df.columns if c not in ["report_id", "row_id", first_col, second_col]]

# Extract C-codes from column headers (C0020, C0030, etc.)
import re
column_id_mapping = {}
for col_name in value_columns:
    # Try to extract C-code from column name (e.g., "C0020" or "Medical ex C0020")
    match = re.search(r'(C\d+)', col_name)
    if match:
        column_id_mapping[col_name] = match.group(1)
    else:
        # If no C-code found in header, use the column name as is
        column_id_mapping[col_name] = col_name

print(f"\nValue columns to unpivot: {value_columns}")
print(f"Column ID mapping: {column_id_mapping}")

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

# Create struct for each column using the mapped column_ids
struct_cols = [
    struct(lit(column_id_mapping[c]).alias("column_id"), col(c).alias("value")) 
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
