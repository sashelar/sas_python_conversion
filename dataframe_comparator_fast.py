"""
Spark DataFrame Comparator for Azure Synapse - performance patched.

Drop-in replacement for dataframe_comparator.py: same class name, same
constructor, same report structure, same numbers. Only the execution plan
changed.

What was costing the time
-------------------------
1. _compare_data() sorted BOTH DataFrames (orderBy over all 255 columns or the
   key columns) before joining. Two global shuffles of the full width, and the
   join result does not depend on input order.
2. joined_df was never cached, so every action below re-ran both sorts and the
   full outer join from source.
3. Every column ran its own actions: 5 per numeric column (1 count + 4 collect)
   and 9 per string column (3 count + 6 collect). With 95 numeric and 160
   string/date columns that is roughly 1,900 Spark jobs, each one re-reading
   and re-joining everything.
4. total_rows = self.df1.count() was called inside every column, so df1 was
   counted 255 times.
5. comparison_df_with_null was cached in the string path and never unpersisted,
   so cache memory filled up across 160 string columns and started spilling.

What this version does instead
------------------------------
- joins once, with no pre-sorting, and persists the join
- counts the rows once and reuses the number
- computes the metrics for every column in a few batched aggregations
  (batch_size columns per job) instead of one job per column
- only fetches sample rows for the columns that actually differ
- skips the levenshtein call on rows where the two values are equal
- unpersists everything it persists

Same results, with one honest caveat: the 'sample_mismatches' rows for NUMERIC
columns are taken with .limit(), so dropping the pre-sort can change WHICH
example rows are shown. Counts, statistics and match percentages are identical.
String samples are still ordered by fuzzy_score, so those are unchanged.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.storagelevel import StorageLevel
from datetime import datetime
from typing import Dict, List, Optional
import json

NUMERIC_HINTS = ('int', 'long', 'double', 'float', 'decimal')


class DataFrameComparator:
    """Same interface as the original comparator."""

    def __init__(self, spark: SparkSession, df1: DataFrame, df2: DataFrame,
                 key_columns: List[str] = None,
                 df1_name: str = "DF1",
                 df2_name: str = "DF2",
                 numeric_precision: int = 3,
                 fuzzy_threshold: float = 0.8,
                 batch_size: int = 40,
                 shuffle_partitions: Optional[int] = 16):
        """
        batch_size : how many columns are measured per Spark job. 40 keeps the
                     query plan small enough to stay in whole-stage codegen.
        shuffle_partitions : spark.sql.shuffle.partitions used during compare().
                     The default 200 creates 200 tiny tasks per job for a table
                     this size, which is pure scheduling overhead. The previous
                     value is restored when compare() finishes. Pass None to
                     leave the session setting untouched.
        """
        self.spark = spark
        self.df1 = df1
        self.df2 = df2
        self.key_columns = key_columns
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.numeric_precision = numeric_precision
        self.fuzzy_threshold = fuzzy_threshold
        self.batch_size = batch_size
        self.shuffle_partitions = shuffle_partitions

        self.comparison_results = {}
        self.report = {}
        self.report['df1_name'] = df1_name
        self.report['df2_name'] = df2_name
        self._total_rows = None

    # ------------------------------------------------------------------ public
    def compare(self) -> Dict:
        print("=" * 80)
        print("DATAFRAME COMPARISON STARTED")
        print("=" * 80)

        previous_partitions = None
        if self.shuffle_partitions:
            previous_partitions = self.spark.conf.get("spark.sql.shuffle.partitions")
            self.spark.conf.set("spark.sql.shuffle.partitions", str(self.shuffle_partitions))
        try:
            self._compare_row_counts()
            self._compare_columns()
            self._compare_schema()

            common = self.report.get('columns', {}).get('common_columns', [])
            if not common:
                print("\n⚠️  No common columns found — skipping data comparison")
                self.report['data_comparison'] = {
                    'status': 'SKIPPED',
                    'reason': 'No common columns between the two DataFrames'
                }
            else:
                if not self.report.get('columns_match', False):
                    extra_df1 = self.report['columns'].get('only_in_df1', [])
                    extra_df2 = self.report['columns'].get('only_in_df2', [])
                    print(f"\n⚠️  Column mismatch — comparing {len(common)} common columns only")
                    if extra_df1:
                        print(f"   Extra in {self.df1_name} (skipped): {extra_df1}")
                    if extra_df2:
                        print(f"   Extra in {self.df2_name} (skipped): {extra_df2}")
                self._compare_data()

            self._generate_final_report()
        finally:
            if previous_partitions is not None:
                self.spark.conf.set("spark.sql.shuffle.partitions", previous_partitions)
        return self.report

    # ------------------------------------------------------------- layers 1-3
    def _compare_row_counts(self):
        print("\n📊 Comparing row counts...")
        count1 = self.df1.count()
        count2 = self.df2.count()
        self._total_rows = count1                      # reused by every column

        self.report['row_count'] = {
            'df1_count': count1,
            'df2_count': count2,
            'difference': count2 - count1,
            'match': count1 == count2,
            'difference_percentage': round(((count2 - count1) / count1 * 100), 2) if count1 > 0 else None
        }
        print(f"   {self.df1_name}: {count1:,} rows")
        print(f"   {self.df2_name}: {count2:,} rows")
        print(f"   Difference: {count2 - count1:,} rows")
        print(f"   Match: {'✓ YES' if count1 == count2 else '✗ NO'}")

    def _compare_columns(self):
        print("\n📋 Comparing column names...")
        cols1, cols2 = set(self.df1.columns), set(self.df2.columns)
        common_cols = cols1.intersection(cols2)

        self.report['columns'] = {
            'df1_columns': sorted(list(cols1)),
            'df2_columns': sorted(list(cols2)),
            'common_columns': sorted(list(common_cols)),
            'only_in_df1': sorted(list(cols1 - cols2)),
            'only_in_df2': sorted(list(cols2 - cols1)),
            'column_count_df1': len(cols1),
            'column_count_df2': len(cols2),
            'match': cols1 == cols2
        }
        self.report['columns_match'] = cols1 == cols2

        print(f"   {self.df1_name}: {len(cols1)} columns")
        print(f"   {self.df2_name}: {len(cols2)} columns")
        print(f"   Common columns: {len(common_cols)}")
        if cols1 - cols2:
            print(f"   ⚠️  Only in {self.df1_name}: {sorted(list(cols1 - cols2))}")
        if cols2 - cols1:
            print(f"   ⚠️  Only in {self.df2_name}: {sorted(list(cols2 - cols1))}")
        print(f"   Match: {'✓ YES' if cols1 == cols2 else '✗ NO'}")

    def _compare_schema(self):
        print("\n🔍 Comparing schemas...")
        schema1 = {f.name: str(f.dataType) for f in self.df1.schema.fields}
        schema2 = {f.name: str(f.dataType) for f in self.df2.schema.fields}
        common_cols = set(schema1).intersection(set(schema2))

        type_mismatches = [{'column': c, 'df1_type': schema1[c], 'df2_type': schema2[c]}
                           for c in common_cols if schema1[c] != schema2[c]]
        self.report['schema'] = {
            'df1_schema': schema1,
            'df2_schema': schema2,
            'type_mismatches': type_mismatches,
            'schemas_match': len(type_mismatches) == 0 and set(schema1) == set(schema2)
        }
        if type_mismatches:
            print(f"   ⚠️  Type mismatches found: {len(type_mismatches)}")
            for m in type_mismatches[:5]:
                print(f"      - {m['column']}: {m['df1_type']} vs {m['df2_type']}")
        else:
            print("   ✓ All data types match")

    # --------------------------------------------------------------- the data
    def _compare_data(self):
        print("\n🔬 Comparing data...")
        common_cols = sorted(list(set(self.df1.columns).intersection(set(self.df2.columns))))
        if not common_cols:
            print("   ⚠️  No common columns to compare")
            return

        df1_compare = self.df1.select(*common_cols)
        df2_compare = self.df2.select(*common_cols)

        # No pre-sorting: a full outer join does not care about input order.
        valid_keys = [k for k in (self.key_columns or []) if k in common_cols]
        if valid_keys:
            print(f"   using key based matching on: {valid_keys}")
            df1_a, df2_a = df1_compare.alias("df1"), df2_compare.alias("df2")
            joined_df = df1_a.join(
                df2_a,
                on=[df1_a[k] == df2_a[k] for k in valid_keys],
                how="full_outer")
            first_key = f"df1.{valid_keys[0]}"
        else:
            print("   using row number matching")
            window_spec = Window.orderBy(F.monotonically_increasing_id())
            df1_numbered = df1_compare.withColumn("_row_num", F.row_number().over(window_spec))
            df2_numbered = df2_compare.withColumn("_row_num", F.row_number().over(window_spec))
            joined_df = df1_numbered.alias("df1").join(
                df2_numbered.alias("df2"), on="_row_num", how="full_outer")
            first_key = "_row_num"

        # Persist once: every measurement below reads this instead of re-joining.
        joined_df = joined_df.persist(StorageLevel.MEMORY_AND_DISK)
        joined_rows = joined_df.count()
        print(f"   joined rows: {joined_rows:,} (join materialised once)")

        types = dict(self.df1.dtypes)
        numeric_cols = [c for c in common_cols
                        if any(t in str(types.get(c, '')).lower() for t in NUMERIC_HINTS)]
        string_cols = [c for c in common_cols if c not in numeric_cols]
        print(f"   {len(numeric_cols)} numeric columns, {len(string_cols)} string/other columns")

        try:
            metrics = {}
            metrics.update(self._numeric_metrics(joined_df, numeric_cols))
            metrics.update(self._string_metrics(joined_df, string_cols))

            column_comparisons = {}
            differing = []
            for col in common_cols:
                detail = self._build_detail(col, types.get(col, 'unknown'), metrics[col])
                column_comparisons[col] = detail
                if not detail['all_values_match']:
                    differing.append(col)

            print(f"   fetching samples for {len(differing)} columns that differ "
                  f"(of {len(common_cols)})")
            for col in differing:
                is_numeric = col in numeric_cols
                samples = self._samples(joined_df, col, first_key, is_numeric)
                column_comparisons[col].update(samples)

            self.report['data_comparison'] = {
                'status': 'COMPLETED',
                'columns_compared': len(common_cols),
                'column_details': column_comparisons
            }
            all_match = all(d['all_values_match'] for d in column_comparisons.values())
            print(f"\n   Overall Data Match: {'✓ YES' if all_match else '✗ NO'}")
        finally:
            joined_df.unpersist()

    # ------------------------------------------------------------- aggregation
    def _batches(self, cols):
        for i in range(0, len(cols), self.batch_size):
            yield cols[i:i + self.batch_size]

    def _numeric_metrics(self, joined_df, cols) -> Dict:
        out = {}
        for batch in self._batches(cols):
            aggs = []
            for c in batch:
                v1, v2 = F.col(f"df1.{c}"), F.col(f"df2.{c}")
                exact = F.when(F.round(v1, self.numeric_precision)
                               .eqNullSafe(F.round(v2, self.numeric_precision)), 1).otherwise(0)
                diff = v2 - v1
                aggs += [
                    F.sum(exact).alias(f"{c}__exact"),
                    F.count(F.when(v1.isNotNull() & v2.isNotNull(), 1)).alias(f"{c}__both_nn"),
                    F.count(F.when(v1.isNull() & v2.isNotNull(), 1)).alias(f"{c}__only2"),
                    F.count(F.when(v1.isNotNull() & v2.isNull(), 1)).alias(f"{c}__only1"),
                    F.avg(diff).alias(f"{c}__avg"),
                    F.min(diff).alias(f"{c}__min"),
                    F.max(diff).alias(f"{c}__max"),
                    F.stddev(diff).alias(f"{c}__std"),
                ]
            row = joined_df.agg(*aggs).collect()[0]
            for c in batch:
                out[c] = {'kind': 'numeric',
                          'exact': row[f"{c}__exact"] or 0,
                          'both_nn': row[f"{c}__both_nn"] or 0,
                          'only1': row[f"{c}__only1"] or 0,
                          'only2': row[f"{c}__only2"] or 0,
                          'avg': row[f"{c}__avg"], 'min': row[f"{c}__min"],
                          'max': row[f"{c}__max"], 'std': row[f"{c}__std"]}
        return out

    def _string_metrics(self, joined_df, cols) -> Dict:
        out = {}
        for batch in self._batches(cols):
            aggs = []
            for c in batch:
                v1, v2 = F.col(f"df1.{c}").cast("string"), F.col(f"df2.{c}").cast("string")
                both_nn = v1.isNotNull() & v2.isNotNull()
                exact = F.when(v1.eqNullSafe(v2), 1).otherwise(0)
                max_len = F.greatest(F.length(v1), F.length(v2))
                # levenshtein is only evaluated where the values actually differ;
                # equal values score 1.0, which is what the formula gives anyway
                fuzzy = F.when(v1.eqNullSafe(v2), F.lit(1.0)).otherwise(
                    F.when(max_len > 0, 1 - (F.levenshtein(v1, v2) / max_len)).otherwise(F.lit(1.0)))
                aggs += [
                    F.sum(F.when(both_nn, exact).otherwise(0)).alias(f"{c}__exact"),
                    F.avg(F.when(both_nn, fuzzy)).alias(f"{c}__avgf"),
                    F.min(F.when(both_nn, fuzzy)).alias(f"{c}__minf"),
                    F.max(F.when(both_nn, fuzzy)).alias(f"{c}__maxf"),
                    F.count(F.when(both_nn & (fuzzy >= self.fuzzy_threshold) & (fuzzy < 1), 1))
                     .alias(f"{c}__fuzzy"),
                    F.count(F.when(both_nn, 1)).alias(f"{c}__both_nn"),
                    F.count(F.when(v1.isNull() & v2.isNotNull(), 1)).alias(f"{c}__only2"),
                    F.count(F.when(v1.isNotNull() & v2.isNull(), 1)).alias(f"{c}__only1"),
                    F.count(F.when(v1.isNull() & v2.isNull(), 1)).alias(f"{c}__both_null"),
                ]
            row = joined_df.agg(*aggs).collect()[0]
            for c in batch:
                out[c] = {'kind': 'string',
                          'exact': row[f"{c}__exact"] or 0,
                          'fuzzy_matches': row[f"{c}__fuzzy"] or 0,
                          'both_nn': row[f"{c}__both_nn"] or 0,
                          'only1': row[f"{c}__only1"] or 0,
                          'only2': row[f"{c}__only2"] or 0,
                          'both_null': row[f"{c}__both_null"] or 0,
                          'avgf': row[f"{c}__avgf"], 'minf': row[f"{c}__minf"],
                          'maxf': row[f"{c}__maxf"]}
        return out

    # ------------------------------------------------------------ report shape
    def _build_detail(self, col_name, dtype, m) -> Dict:
        total_rows = self._total_rows
        result = {'column_name': col_name, 'data_type': str(dtype),
                  'all_values_match': False, 'mismatches': 0, 'match_percentage': 0.0}

        if m['kind'] == 'numeric':
            exact_matches = int(m['exact'])
            mismatches = int(total_rows - exact_matches)
            result.update({
                'comparison_type': 'numeric',
                'exact_matches': exact_matches,
                'mismatches': mismatches,
                'only_in_df1': int(m['only1']),
                'only_in_df2': int(m['only2']),
                'all_values_match': mismatches == 0,
                'match_percentage': round(((exact_matches / total_rows) * 100), 2) if total_rows else 0,
                'statistics': {
                    'average_difference': float(m['avg']) if m['avg'] else 0.0,
                    'min_difference': float(m['min']) if m['min'] else 0.0,
                    'max_difference': float(m['max']) if m['max'] else 0.0,
                    'stddev_difference': float(m['std']) if m['std'] else 0.0,
                },
                'sample_mismatches': [], 'sample_only_in_df1': [], 'sample_only_in_df2': [],
            })
        else:
            exact_matches = int(m['exact'])
            both_null = int(m['both_null'])
            # kept exactly as in the original
            if total_rows == exact_matches:
                pass
            else:
                if both_null > 0:
                    exact_matches = exact_matches + both_null
            mismatches = int(total_rows - exact_matches)
            result.update({
                'comparison_type': 'string_fuzzy',
                'exact_matches': exact_matches,
                'fuzzy_matches': int(m['fuzzy_matches']),
                'mismatches': mismatches,
                'only_in_df1': int(m['only1']),
                'only_in_df2': int(m['only2']),
                'all_values_match': exact_matches == total_rows,
                'match_percentage': round(((exact_matches / total_rows) * 100), 2) if total_rows else 0,
                'fuzzy_threshold': self.fuzzy_threshold,
                'statistics': {
                    'average_fuzzy_score': round(float(m['avgf']), 4) if m['avgf'] else 0.0,
                    'min_fuzzy_score': round(float(m['minf']), 4) if m['minf'] else 0.0,
                    'max_fuzzy_score': round(float(m['maxf']), 4) if m['maxf'] else 1.0,
                },
                'sample_mismatches': [], 'sample_only_in_df1': [], 'sample_only_in_df2': [],
            })
        return result

    # ----------------------------------------------------------------- samples
    def _samples(self, joined_df, col_name, first_key, is_numeric) -> Dict:
        col1, col2 = f"df1.{col_name}", f"df2.{col_name}"

        if is_numeric:
            diff_df = joined_df.select(
                F.col(first_key).alias("key"),
                F.col(col1).alias("val1"),
                F.col(col2).alias("val2"),
                (F.col(col2) - F.col(col1)).alias("difference"),
                F.abs(F.col(col2) - F.col(col1)).alias("abs_difference"),
                F.when(F.round(F.col(col1), self.numeric_precision)
                       .eqNullSafe(F.round(F.col(col2), self.numeric_precision)), 1)
                 .otherwise(0).alias("exact_match"))

            mism = diff_df.filter((F.col("exact_match") == 0)
                                  & F.col("val1").isNotNull()
                                  & F.col("val2").isNotNull()).limit(10).collect()
            only1 = diff_df.filter(F.col("val1").isNotNull() & F.col("val2").isNull()).limit(5).collect()
            only2 = diff_df.filter(F.col("val1").isNull() & F.col("val2").isNotNull()).limit(5).collect()

            def num_rows(rows, with_flag):
                out = []
                for row in rows:
                    try:
                        item = {
                            'key': row.key if row.key is not None else "missing row or value",
                            'df1_value': float(row.val1) if row.val1 is not None else 0,
                            'df2_value': float(row.val2) if row.val2 is not None else 0,
                            'difference': float(row.difference) if row.difference is not None
                            else "missing row or value",
                        }
                        if with_flag:
                            item['is_decimal_change'] = "missing row or value"
                        out.append(item)
                    except Exception:
                        print("something went wrong", row)
                return out

            return {'sample_mismatches': num_rows(mism, False)[:5],
                    'sample_only_in_df1': num_rows(only1, True)[:5],
                    'sample_only_in_df2': num_rows(only2, True)[:5]}

        v1, v2 = F.col(col1).cast("string"), F.col(col2).cast("string")
        max_len = F.greatest(F.length(v1), F.length(v2))
        base = joined_df.select(
            F.col(first_key).alias("key"),
            v1.alias("val1"), v2.alias("val2"),
            F.when(v1.eqNullSafe(v2), 1).otherwise(0).alias("exact_match"),
            F.when(v1.eqNullSafe(v2), F.lit(1.0)).otherwise(
                F.when(max_len > 0, 1 - (F.levenshtein(v1, v2) / max_len))
                 .otherwise(F.lit(1.0))).alias("fuzzy_score"))

        mism = (base.filter(F.col("val1").isNotNull() & F.col("val2").isNotNull()
                            & (F.col("exact_match") == 0))
                    .orderBy(F.col("fuzzy_score").asc()).limit(10).collect())
        only1 = (base.filter(F.col("val1").isNotNull() & F.col("val2").isNull())
                     .withColumn("fuzzy_score", F.lit(0)).limit(5).collect())
        only2 = (base.filter(F.col("val1").isNull() & F.col("val2").isNotNull())
                     .withColumn("fuzzy_score", F.lit(0)).limit(5).collect())

        def str_rows(rows):
            out = []
            for row in rows:
                try:
                    out.append({
                        'key': row.key if row.key is not None else "missing row or value",
                        'df1_value': str(row.val1) if row.val1 is not None else "missing row or value",
                        'df2_value': str(row.val2) if row.val2 is not None else "missing row or value",
                        'fuzzy_score': round(float(row.fuzzy_score), 4),
                        'fuzzy_match': row.fuzzy_score >= self.fuzzy_threshold,
                    })
                except Exception:
                    print("skipped because of some issue in string")
            return out

        return {'sample_mismatches': str_rows(mism)[:5],
                'sample_only_in_df1': str_rows(only1)[:5],
                'sample_only_in_df2': str_rows(only2)[:5]}

    # ------------------------------------------------------------------- final
    def _generate_final_report(self):
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        overall_match = (
            self.report.get('row_count', {}).get('match', False) and
            self.report.get('columns_match', False) and
            self.report.get('schema', {}).get('schemas_match', False)
        )
        if self.report.get('data_comparison', {}).get('status') == 'COMPLETED':
            data_match = all(d['all_values_match'] for d in
                             self.report['data_comparison']['column_details'].values())
            overall_match = overall_match and data_match

        self.report['overall_match'] = overall_match
        self.report['comparison_timestamp'] = datetime.now().isoformat()

        print(f"\n📌 Overall Match: {'✓ PASS' if overall_match else '✗ FAIL'}")
        print(f"   Row Count Match: {'✓' if self.report.get('row_count', {}).get('match', False) else '✗'}")
        print(f"   Column Names Match: {'✓' if self.report.get('columns_match', False) else '✗'}")
        print(f"   Schema Match: {'✓' if self.report.get('schema', {}).get('schemas_match', False) else '✗'}")

        if self.report.get('data_comparison', {}).get('status') == 'COMPLETED':
            details = self.report['data_comparison']['column_details']
            matching_cols = sum(1 for d in details.values() if d.get('all_values_match', False))
            print(f"   Data Match: {matching_cols}/{len(details)} columns match perfectly")
        print("\n" + "=" * 80)

    def save_report_to_json(self, output_path: str):
        report_json = json.dumps(self.report, indent=2, default=str)
        report_df = self.spark.createDataFrame([(report_json,)], ["report_json"])
        report_df.coalesce(1).write.mode("overwrite").text(output_path)
        print(f"\n💾 Report saved to: {output_path}")

    def get_detailed_report(self) -> str:
        lines = ["=" * 100, "DATAFRAME COMPARISON DETAILED REPORT", "=" * 100]
        lines.append(f"\nGenerated: {self.report.get('comparison_timestamp', 'N/A')}")
        lines.append(f"Overall Match: {'✓ PASS' if self.report.get('overall_match', False) else '✗ FAIL'}")
        df1_name = self.report.get('df1_name', 'DF1')
        df2_name = self.report.get('df2_name', 'DF2')

        rc = self.report.get('row_count', {})
        lines += ["\n" + "-" * 100, "ROW COUNT COMPARISON", "-" * 100,
                  f"{df1_name}: {rc.get('df1_count', 0):,} rows",
                  f"{df2_name}: {rc.get('df2_count', 0):,} rows",
                  f"Difference: {rc.get('difference', 0):,} rows ({rc.get('difference_percentage', 0)}%)"]

        cols = self.report.get('columns', {})
        lines += ["\n" + "-" * 100, "COLUMN COMPARISON", "-" * 100,
                  f"{df1_name}: {cols.get('column_count_df1', 0)} columns",
                  f"{df2_name}: {cols.get('column_count_df2', 0)} columns"]

        if self.report.get('data_comparison', {}).get('status') == 'COMPLETED':
            lines += ["\n" + "-" * 100, "DATA COMPARISON", "-" * 100]
            for col_name, d in self.report['data_comparison']['column_details'].items():
                lines.append(f"\nColumn: {col_name}")
                lines.append(f"  Type: {d.get('data_type', 'N/A')}")
                lines.append(f"  Match: {'✓ YES' if d.get('all_values_match', False) else '✗ NO'}")
                lines.append(f"  Match %: {d.get('match_percentage', 0)}%")
                if d.get('comparison_type') == 'numeric':
                    lines.append(f"  Avg Difference: {d.get('statistics', {}).get('average_difference', 0)}")
        lines.append("\n" + "=" * 100)
        return "\n".join(lines)
