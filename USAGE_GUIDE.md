# DataFrameComparator - Usage Guide for Azure Synapse

## Overview
This guide demonstrates how to use the DataFrameComparator class in Azure Synapse to compare Spark DataFrames loaded from ADLS (Azure Data Lake Storage).

## Features
✅ Row count comparison with percentage difference
✅ Column name matching
✅ Schema/data type validation
✅ Numeric comparison with difference calculation (includes decimal changes)
✅ String comparison with fuzzy matching (0-1 score)
✅ Automatic sorting for consistent comparison
✅ No dbutils dependency - pure Spark
✅ Works directly with ADLS paths
✅ Detailed reporting with sample mismatches

---

## Installation
Simply copy the `dataframe_comparator.py` file to your Synapse workspace or execute it as a notebook cell.

---

## Basic Usage

### Scenario 1: Compare Two Parquet Files from ADLS

```python
from pyspark.sql import SparkSession
from dataframe_comparator import DataFrameComparator

# Get Spark session (already available in Synapse)
spark = spark  # or SparkSession.builder.getOrCreate()

# Load DataFrames from ADLS
df1 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/source/employees.parquet")
df2 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/target/employees.parquet")

# Initialize comparator with key columns for row matching
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['employee_id'],  # Primary key for row matching
    numeric_precision=6,           # Decimal precision
    fuzzy_threshold=0.85           # 85% similarity for string matching
)

# Run comparison
report = comparator.compare()

# Display detailed report
print(comparator.get_detailed_report())
```

---

### Scenario 2: Compare CSV Files with Custom Settings

```python
# Load CSV files from ADLS
df1 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/sales_2023.csv"
)
df2 = spark.read.option("header", "true").option("inferSchema", "true").csv(
    "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/sales_2024.csv"
)

# Compare with composite key
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['region', 'product_id', 'date'],  # Composite key
    numeric_precision=2,    # 2 decimal places for currency
    fuzzy_threshold=0.9     # Strict string matching
)

report = comparator.compare()
```

---

### Scenario 3: Compare Delta Tables

```python
# Read Delta tables
df1 = spark.read.format("delta").load(
    "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/delta/customers_v1"
)
df2 = spark.read.format("delta").load(
    "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/delta/customers_v2"
)

# Compare
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['customer_id']
)

report = comparator.compare()

# Save report to ADLS
comparator.save_report_to_json(
    "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/reports/customer_comparison.json"
)
```

---

### Scenario 4: Compare Without Key Columns (Row-by-Row)

When you don't have a natural key, the comparator will sort by all columns and compare row-by-row:

```python
df1 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/transactions_old.parquet")
df2 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/transactions_new.parquet")

# No key_columns specified - will use row-by-row comparison after sorting
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=None  # Will sort by all columns
)

report = comparator.compare()
```

---

### Scenario 5: Filtering Data Before Comparison

```python
# Load full datasets
df1_full = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/orders_2023.parquet")
df2_full = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/data/orders_2024.parquet")

# Filter to specific region before comparison
df1 = df1_full.filter("region = 'NORTH_AMERICA'")
df2 = df2_full.filter("region = 'NORTH_AMERICA'")

comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['order_id']
)

report = comparator.compare()
```

---

## Understanding the Report

### Report Structure

The comparison report contains the following sections:

```python
{
    "overall_match": True/False,
    "comparison_timestamp": "2024-01-15T10:30:00",
    
    "row_count": {
        "df1_count": 10000,
        "df2_count": 10005,
        "difference": 5,
        "difference_percentage": 0.05,
        "match": False
    },
    
    "columns": {
        "df1_columns": ["id", "name", "age"],
        "df2_columns": ["id", "name", "age", "email"],
        "common_columns": ["id", "name", "age"],
        "only_in_df1": [],
        "only_in_df2": ["email"],
        "match": False
    },
    
    "schema": {
        "type_mismatches": [
            {
                "column": "age",
                "df1_type": "IntegerType",
                "df2_type": "LongType"
            }
        ],
        "schemas_match": False
    },
    
    "data_comparison": {
        "status": "COMPLETED",
        "column_details": {
            "salary": {
                "comparison_type": "numeric",
                "all_values_match": False,
                "match_percentage": 98.5,
                "statistics": {
                    "average_difference": 0.15,
                    "min_difference": -100.0,
                    "max_difference": 250.0
                },
                "sample_mismatches": [
                    {
                        "df1_value": 50000.50,
                        "df2_value": 50000.51,
                        "difference": 0.01,
                        "is_decimal_change": True
                    }
                ]
            },
            "name": {
                "comparison_type": "string_fuzzy",
                "all_values_match": False,
                "match_percentage": 99.2,
                "fuzzy_threshold": 0.85,
                "statistics": {
                    "average_fuzzy_score": 0.9845,
                    "min_fuzzy_score": 0.75
                },
                "sample_mismatches": [
                    {
                        "df1_value": "Jane Smith",
                        "df2_value": "Jane Smyth",
                        "fuzzy_score": 0.9091,
                        "fuzzy_match": True
                    }
                ]
            }
        }
    }
}
```

---

## Advanced Examples

### Example 1: Comparing with Data Transformations

```python
# Load raw data
df1_raw = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/raw/product_data_v1.parquet")
df2_raw = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/raw/product_data_v2.parquet")

# Apply transformations before comparison
from pyspark.sql.functions import col, upper, trim

df1 = df1_raw.select(
    col("product_id"),
    upper(trim(col("product_name"))).alias("product_name"),
    col("price"),
    col("category")
)

df2 = df2_raw.select(
    col("product_id"),
    upper(trim(col("product_name"))).alias("product_name"),
    col("price"),
    col("category")
)

# Compare normalized data
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['product_id'],
    fuzzy_threshold=0.95  # Strict matching after normalization
)

report = comparator.compare()
```

---

### Example 2: Comparing Aggregated Data

```python
from pyspark.sql.functions import sum, avg, count

# Load transaction data
transactions_2023 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/transactions/2023.parquet")
transactions_2024 = spark.read.parquet("abfss://mycontainer@mystorageaccount.dfs.core.windows.net/transactions/2024.parquet")

# Aggregate by region
df1 = transactions_2023.groupBy("region").agg(
    sum("amount").alias("total_amount"),
    avg("amount").alias("avg_amount"),
    count("*").alias("transaction_count")
)

df2 = transactions_2024.groupBy("region").agg(
    sum("amount").alias("total_amount"),
    avg("amount").alias("avg_amount"),
    count("*").alias("transaction_count")
)

# Compare aggregated results
comparator = DataFrameComparator(
    spark=spark,
    df1=df1,
    df2=df2,
    key_columns=['region'],
    numeric_precision=2
)

report = comparator.compare()
```

---

### Example 3: Running Multiple Comparisons in a Loop

```python
# Define list of tables to compare
tables_to_compare = [
    ("employees", "employee_id"),
    ("departments", "dept_id"),
    ("projects", "project_id")
]

base_path_old = "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/old_data"
base_path_new = "abfss://mycontainer@mystorageaccount.dfs.core.windows.net/new_data"

all_reports = {}

for table_name, key_col in tables_to_compare:
    print(f"\n{'='*80}")
    print(f"Comparing {table_name}...")
    print(f"{'='*80}")
    
    # Load data
    df1 = spark.read.parquet(f"{base_path_old}/{table_name}.parquet")
    df2 = spark.read.parquet(f"{base_path_new}/{table_name}.parquet")
    
    # Compare
    comparator = DataFrameComparator(
        spark=spark,
        df1=df1,
        df2=df2,
        key_columns=[key_col]
    )
    
    report = comparator.compare()
    all_reports[table_name] = report
    
    # Save individual report
    comparator.save_report_to_json(
        f"abfss://mycontainer@mystorageaccount.dfs.core.windows.net/reports/{table_name}_comparison.json"
    )

# Check overall results
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
for table_name, report in all_reports.items():
    status = "✓ PASS" if report['overall_match'] else "✗ FAIL"
    print(f"{table_name}: {status}")
```

---

## Interpreting Results

### Numeric Comparisons

For numeric columns, the comparator calculates:
- **Exact matches**: Values that are exactly equal
- **Average difference**: Mean of (df2_value - df1_value)
- **Min/Max difference**: Range of differences
- **Decimal changes**: Flags differences less than 1.0

**Example interpretation:**
```
Column: salary
  Match %: 98.5%
  Avg Difference: 0.15
  Sample: DF1: 50000.50 | DF2: 50000.51 | Diff: 0.01 (decimal change)
```
This shows salary values are 98.5% identical, with an average increase of $0.15, and the sample shows a small decimal-level change.

---

### String Comparisons

For string columns, the comparator uses:
- **Fuzzy score**: 0 to 1, where 1 is exact match
- **Levenshtein distance**: Edit distance between strings
- **Fuzzy threshold**: Configurable threshold for "close enough"

**Example interpretation:**
```
Column: customer_name
  Match %: 99.2%
  Avg Fuzzy Score: 0.9845
  Sample: DF1: 'Jane Smith' | DF2: 'Jane Smyth' | Score: 0.9091
```
This shows 99.2% of names match exactly, and even mismatches have high similarity (0.91 for "Smith" vs "Smyth").

---

## Best Practices

### 1. Choose Appropriate Key Columns
```python
# Good: Use natural primary key
key_columns=['customer_id']

# Good: Use composite key when needed
key_columns=['region', 'product_id', 'date']

# Okay: Let it sort by all columns if no natural key
key_columns=None
```

### 2. Set Precision Based on Data Type
```python
# Currency: 2 decimal places
numeric_precision=2

# Scientific data: 6 decimal places
numeric_precision=6

# Integer comparisons: 0 decimal places
numeric_precision=0
```

### 3. Adjust Fuzzy Threshold
```python
# Strict matching (names, IDs)
fuzzy_threshold=0.95

# Moderate matching (addresses, descriptions)
fuzzy_threshold=0.85

# Lenient matching (user-generated content)
fuzzy_threshold=0.75
```

### 4. Filter Large Datasets
```python
# Compare only recent data
df1 = df1.filter("transaction_date >= '2024-01-01'")
df2 = df2.filter("transaction_date >= '2024-01-01'")
```

### 5. Cache DataFrames for Multiple Operations
```python
df1.cache()
df2.cache()

# Run multiple comparisons with different settings
comparator1 = DataFrameComparator(spark, df1, df2, fuzzy_threshold=0.8)
comparator2 = DataFrameComparator(spark, df1, df2, fuzzy_threshold=0.9)
```

---

## Troubleshooting

### Issue: Memory errors with large DataFrames
**Solution**: Filter or sample data before comparison
```python
# Sample 10% of data
df1_sample = df1.sample(fraction=0.1, seed=42)
df2_sample = df2.sample(fraction=0.1, seed=42)
```

### Issue: Slow comparison performance
**Solution**: Ensure data is partitioned appropriately
```python
# Repartition before comparison
df1 = df1.repartition(100, "key_column")
df2 = df2.repartition(100, "key_column")
```

### Issue: Type mismatches preventing comparison
**Solution**: Cast columns to compatible types
```python
from pyspark.sql.functions import col
df2 = df2.withColumn("age", col("age").cast("integer"))
```

---

## Complete Working Example for Azure Synapse

```python
# ================================================
# Complete Example: Employee Data Comparison
# ================================================

from pyspark.sql import SparkSession
from dataframe_comparator import DataFrameComparator

# Initialize Spark (already available in Synapse)
spark = spark

# Define ADLS paths
STORAGE_ACCOUNT = "mystorageaccount"
CONTAINER = "mycontainer"
BASE_PATH = f"abfss://{CONTAINER}@{STORAGE_ACCOUNT}.dfs.core.windows.net"

# Load source and target employee data
source_path = f"{BASE_PATH}/hr/source/employees.parquet"
target_path = f"{BASE_PATH}/hr/target/employees.parquet"

print(f"Loading source data from: {source_path}")
df_source = spark.read.parquet(source_path)

print(f"Loading target data from: {target_path}")
df_target = spark.read.parquet(target_path)

# Show sample data
print("\nSource Data Sample:")
df_source.show(5, truncate=False)

print("\nTarget Data Sample:")
df_target.show(5, truncate=False)

# Initialize comparator
comparator = DataFrameComparator(
    spark=spark,
    df1=df_source,
    df2=df_target,
    key_columns=['employee_id'],
    numeric_precision=2,
    fuzzy_threshold=0.85
)

# Run comparison
print("\nRunning comparison...")
report = comparator.compare()

# Display detailed report
print("\n" + comparator.get_detailed_report())

# Save report to ADLS
report_path = f"{BASE_PATH}/reports/employee_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
comparator.save_report_to_json(report_path)

print(f"\n✅ Comparison complete! Report saved to: {report_path}")

# Access specific parts of the report programmatically
if not report['overall_match']:
    print("\n⚠️ ATTENTION: Differences found!")
    
    if not report['row_count']['match']:
        print(f"   - Row count mismatch: {report['row_count']['difference']} rows difference")
    
    if not report['columns_match']:
        print(f"   - Column mismatch: {report['columns']['only_in_df1']} missing in DF2")
    
    if 'data_comparison' in report:
        for col, details in report['data_comparison']['column_details'].items():
            if not details['all_values_match']:
                print(f"   - {col}: {details['match_percentage']}% match")
```

---

## API Reference

### DataFrameComparator Class

#### Constructor Parameters
- `spark` (SparkSession): Active Spark session
- `df1` (DataFrame): First DataFrame (source/expected)
- `df2` (DataFrame): Second DataFrame (target/actual)
- `key_columns` (List[str], optional): Columns for row matching
- `numeric_precision` (int, default=6): Decimal precision for numeric comparisons
- `fuzzy_threshold` (float, default=0.8): Threshold for fuzzy string matching (0 to 1)

#### Methods
- `compare()`: Execute full comparison and return report dictionary
- `get_detailed_report()`: Return formatted text report
- `save_report_to_json(output_path)`: Save report to ADLS JSON file

---

## License & Support
This tool is designed for Azure Synapse Analytics environments. For issues or enhancements, please contact your data engineering team.
