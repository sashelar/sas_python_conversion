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
# STEP 2: Find C-codes from row just above R-codes
# ====================================================
from pyspark.sql.functions import regexp_extract, when

# Collect all rows
all_rows = df.collect()
column_id_mapping = {}
c_code_row_index = None
first_r_code_row_index = None

print("Finding R-code rows and C-code row location...")

# First pass: find where R-codes start
for idx, row in enumerate(all_rows):
    second_col_value = str(row[1]) if row[1] else ""
    if "R0" in second_col_value or second_col_value.startswith("R"):
        first_r_code_row_index = idx
        print(f"First R-code row found at index {idx}: {second_col_value}")
        break

# C-codes are in the row just before R-codes start
if first_r_code_row_index and first_r_code_row_index > 0:
    c_code_row_index = first_r_code_row_index - 1
    c_code_row = all_rows[c_code_row_index]
    print(f"C-code row at index {c_code_row_index}")
    
    # Extract C-codes from that specific row
    for col_idx, col_name in enumerate(df.columns):
        cell_value = str(c_code_row[col_idx]).strip() if c_code_row[col_idx] else ""
        if cell_value:
            column_id_mapping[col_name] = cell_value
            print(f"  Column {col_idx} ({col_name}): {cell_value}")

print(f"Column ID mapping: {column_id_mapping}")

# ====================================================
# STEP 3: Extract row_id (R0010, R0020, etc.) from 2nd column
# ====================================================
second_col = df.columns[1]
print(f"\nExtracting row_id from 2nd column: '{second_col}'")

# Create row_id from 2nd column
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
# STEP 4: Remove the C-code row (just the one row)
# ====================================================
df_before_filter = df.count()

# Add row index to identify which row is the C-code row
from pyspark.sql.functions import monotonically_increasing_id

df = df.withColumn("_row_index", monotonically_increasing_id())

# Remove only the C-code row
if c_code_row_index is not None:
    df = df.filter(~(col("_row_index") == c_code_row_index))

# Also remove empty rows
df = df.filter(col("row_id").isNotNull() & (col("row_id") != ""))

# Drop the temporary row index column
df = df.drop("_row_index")

df_after_filter = df.count()

print("\n" + "=" * 80)
print(f"STEP 4: REMOVE C-CODE ROW (index {c_code_row_index})")
print("=" * 80)
print(f"Rows before filter: {df_before_filter}")
print(f"Rows after filter: {df_after_filter}")
df.show(20, truncate=False)

# ====================================================
# STEP 5: Get value columns
# ====================================================
# Exclude first 2 columns and our new columns
first_col = df.columns[0]
second_col = df.columns[1]
value_columns = [c for c in df.columns if c not in ["report_id", "row_id", first_col, second_col]]

print(f"\nValue columns to unpivot: {value_columns}")
print(f"Using column ID mapping from data row: {column_id_mapping}")

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
