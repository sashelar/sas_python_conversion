from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, col, explode, array, struct, coalesce, regexp_extract, when
from pyspark.sql.types import StringType


class ExcelUnpivotTransformer:
    """
    Transforms Excel data from wide format to long format (unpivot).
    
    Usage:
        transformer = ExcelUnpivotTransformer(spark)
        df_result = transformer.transform(
            file_path="abfss://...",
            sheet_name="sheet3",
            output_format="csv"  # or "parquet" or "excel"
            output_path="abfss://..."
        )
    """
    
    def __init__(self, spark_session):
        """Initialize with Spark session"""
        self.spark = spark_session
        self.column_id_mapping = {}
        self.c_code_row_index = None
        self.first_r_code_row_index = None
    
    def transform(self, file_path, sheet_name, output_format="csv", output_path=None):
        """
        Main transformation method
        
        Args:
            file_path: Path to Excel file on Azure (abfss://...)
            sheet_name: Sheet name to process
            output_format: "csv", "parquet", "excel", or list of multiple ["csv", "parquet"]
            output_path: Output location on Azure
        
        Returns:
            DataFrame with unpivoted data
        """
        print("=" * 80)
        print(f"EXCEL UNPIVOT TRANSFORMER - Starting")
        print("=" * 80)
        
        # Step 1: Read Excel
        df = self._read_excel(file_path, sheet_name)
        
        # Step 2: Find C-codes and R-codes locations
        self._find_code_locations(df)
        
        # Step 3: Extract row_id from 2nd column
        df = self._extract_row_ids(df)
        
        # Step 4: Add report_id
        df = df.withColumn("report_id", lit(sheet_name))
        
        print("\n" + "=" * 80)
        print("After adding report_id and extracting row_id")
        print("=" * 80)
        print(f"Total rows: {df.count()}")
        df.show(15, truncate=False)
        
        # Step 5: Remove C-code row
        df = self._remove_c_code_row(df)
        
        # Step 6: Get value columns
        first_col = df.columns[0]
        second_col = df.columns[1]
        value_columns = [c for c in df.columns if c not in ["report_id", "row_id", first_col, second_col]]
        
        print(f"\nValue columns to unpivot: {value_columns}")
        print(f"Column ID mapping: {self.column_id_mapping}")
        
        # Step 7: Cast to String and handle nulls
        for c in value_columns:
            df = df.withColumn(
                c,
                coalesce(col(c).cast(StringType()), lit(" "))
            )
        
        print("\n" + "=" * 80)
        print("After casting to string")
        print("=" * 80)
        print(f"Total rows: {df.count()}")
        df.show(15, truncate=False)
        
        # Step 8: Unpivot
        df_unpivot = self._unpivot(df, value_columns)
        
        print("\n" + "=" * 80)
        print("FINAL UNPIVOTED DATA")
        print("=" * 80)
        print(f"Total rows: {df_unpivot.count()}")
        df_unpivot.show(50, truncate=False)
        
        # Step 9: Write to output
        if output_path:
            if isinstance(output_format, list):
                for fmt in output_format:
                    self._write_output(df_unpivot, fmt, output_path)
            else:
                self._write_output(df_unpivot, output_format, output_path)
        
        return df_unpivot
    
    def _read_excel(self, file_path, sheet_name):
        """Read Excel file"""
        print(f"\nReading Excel: {file_path}")
        print(f"Sheet: {sheet_name}")
        
        df = self.spark.read.format("com.crealytics.spark.excel") \
            .option("header", "true") \
            .option("inferSchema", "false") \
            .option("dataAddress", f"'{sheet_name}'!A1") \
            .load(file_path)
        
        print(f"Total rows: {df.count()}")
        print(f"Columns: {df.columns}")
        df.show(10, truncate=False)
        
        return df
    
    def _find_code_locations(self, df):
        """Find where C-codes and R-codes are located"""
        print("\nFinding C-codes and R-codes locations...")
        
        all_rows = df.collect()
        
        # Find first R-code row
        for idx, row in enumerate(all_rows):
            second_col_value = str(row[1]) if row[1] else ""
            if "R0" in second_col_value or second_col_value.startswith("R"):
                self.first_r_code_row_index = idx
                print(f"First R-code row at index {idx}: {second_col_value}")
                break
        
        # C-codes are in row just before R-codes
        if self.first_r_code_row_index and self.first_r_code_row_index > 0:
            self.c_code_row_index = self.first_r_code_row_index - 1
            c_code_row = all_rows[self.c_code_row_index]
            print(f"C-code row at index {self.c_code_row_index}")
            
            # Extract C-codes
            for col_idx, col_name in enumerate(df.columns):
                cell_value = str(c_code_row[col_idx]).strip() if c_code_row[col_idx] else ""
                if cell_value:
                    self.column_id_mapping[col_name] = cell_value
                    print(f"  Column {col_idx} ({col_name}): {cell_value}")
    
    def _extract_row_ids(self, df):
        """Extract R-codes from 2nd column"""
        second_col = df.columns[1]
        print(f"\nExtracting row_id from 2nd column: '{second_col}'")
        
        df = df.withColumn(
            "row_id",
            when(col(second_col).isNotNull(),
                 regexp_extract(col(second_col), r'(R\d+)', 1)
            ).otherwise(col(second_col))
        )
        
        return df
    
    def _remove_c_code_row(self, df):
        """Remove the C-code row from data"""
        df_before = df.count()
        
        from pyspark.sql.functions import monotonically_increasing_id
        
        df = df.withColumn("_row_index", monotonically_increasing_id())
        
        if self.c_code_row_index is not None:
            df = df.filter(~(col("_row_index") == self.c_code_row_index))
        
        df = df.filter(col("row_id").isNotNull() & (col("row_id") != ""))
        df = df.drop("_row_index")
        
        df_after = df.count()
        
        print(f"\nRemoved C-code row (index {self.c_code_row_index})")
        print(f"Rows before: {df_before}, Rows after: {df_after}")
        
        return df
    
    def _unpivot(self, df, value_columns):
        """Unpivot data"""
        struct_cols = [
            struct(lit(self.column_id_mapping.get(c, c)).alias("column_id"), col(c).alias("value")) 
            for c in value_columns
        ]
        
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
        
        return df_unpivot
    
    def _write_output(self, df, output_format, base_output_path):
        """Write output in specified format"""
        print("\n" + "=" * 80)
        print(f"Writing output as: {output_format.upper()}")
        print("=" * 80)
        
        if output_format.lower() == "csv":
            self._write_csv(df, base_output_path)
        elif output_format.lower() == "parquet":
            self._write_parquet(df, base_output_path)
        elif output_format.lower() == "excel":
            self._write_excel(df, base_output_path)
        else:
            print(f"Unknown format: {output_format}")
    
    def _write_csv(self, df, base_output_path):
        """Write as CSV"""
        output_path = f"{base_output_path}/unpivoted_data.csv"
        
        try:
            df.coalesce(1).write.mode("error").option("header", "true").csv(output_path)
            print(f"✓ New CSV written to: {output_path}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⚠ Path exists, overwriting: {output_path}")
                df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)
                print(f"✓ CSV overwritten")
            else:
                print(f"✗ CSV write failed: {e}")
                raise
    
    def _write_parquet(self, df, base_output_path):
        """Write as Parquet"""
        output_path = f"{base_output_path}/unpivoted_data"
        
        try:
            df.write.mode("error").parquet(output_path)
            print(f"✓ New Parquet written to: {output_path}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⚠ Path exists, overwriting: {output_path}")
                df.write.mode("overwrite").parquet(output_path)
                print(f"✓ Parquet overwritten")
            else:
                print(f"✗ Parquet write failed: {e}")
                raise
    
    def _write_excel(self, df, base_output_path):
        """Write as Excel (XLSX)"""
        # Note: Requires additional library like openpyxl or xlsxwriter
        output_path = f"{base_output_path}/unpivoted_data.xlsx"
        
        try:
            # Convert to pandas and write to Excel
            pandas_df = df.toPandas()
            pandas_df.to_excel(output_path, index=False, sheet_name="Data")
            print(f"✓ Excel written to: {output_path}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⚠ File exists, overwriting: {output_path}")
                pandas_df = df.toPandas()
                pandas_df.to_excel(output_path, index=False, sheet_name="Data")
                print(f"✓ Excel overwritten")
            else:
                print(f"✗ Excel write failed: {e}")
                print("Note: Make sure openpyxl or xlsxwriter is installed")
                raise


# ====================================================
# USAGE EXAMPLE
# ====================================================
if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    
    # Initialize transformer
    transformer = ExcelUnpivotTransformer(spark)
    
    # Example 1: Write as CSV only
    # result = transformer.transform(
    #     file_path="abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx",
    #     sheet_name="sheet3",
    #     output_format="csv",
    #     output_path="abfss://<container>@<storage>.dfs.core.windows.net/<output_path>"
    # )
    
    # Example 2: Write as Parquet only
    # result = transformer.transform(
    #     file_path="abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx",
    #     sheet_name="sheet3",
    #     output_format="parquet",
    #     output_path="abfss://<container>@<storage>.dfs.core.windows.net/<output_path>"
    # )
    
    # Example 3: Write as Excel only
    # result = transformer.transform(
    #     file_path="abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx",
    #     sheet_name="sheet3",
    #     output_format="excel",
    #     output_path="abfss://<container>@<storage>.dfs.core.windows.net/<output_path>"
    # )
    
    # Example 4: Write as multiple formats
    # result = transformer.transform(
    #     file_path="abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx",
    #     sheet_name="sheet3",
    #     output_format=["csv", "parquet", "excel"],
    #     output_path="abfss://<container>@<storage>.dfs.core.windows.net/<output_path>"
    # )
    
    # Example 5: Just get the DataFrame without writing
    # result = transformer.transform(
    #     file_path="abfss://<container>@<storage>.dfs.core.windows.net/<path>/your_file.xlsx",
    #     sheet_name="sheet3"
    # )
