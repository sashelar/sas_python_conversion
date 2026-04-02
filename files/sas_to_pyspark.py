import re
from typing import Optional

class SASToSparkConverter:
    """
    Rule-based SAS to PySpark converter.
    Handles: PROC SQL, DATA Step, PROC MEANS, PROC SORT, MERGE, IF/THEN/ELSE
    """

    def __init__(self):
        self.imports = set()

    # ─────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────
    def convert(self, sas_code: str) -> str:
        self.imports = set()
        sas_code = sas_code.strip()

        blocks = self._split_blocks(sas_code)
        converted_blocks = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            upper = block.upper()

            if upper.startswith("PROC SQL"):
                converted_blocks.append(self._convert_proc_sql(block))
            elif upper.startswith("PROC MEANS") or upper.startswith("PROC SUMMARY"):
                converted_blocks.append(self._convert_proc_means(block))
            elif upper.startswith("PROC SORT"):
                converted_blocks.append(self._convert_proc_sort(block))
            elif upper.startswith("PROC FREQ"):
                converted_blocks.append(self._convert_proc_freq(block))
            elif upper.startswith("PROC PRINT"):
                converted_blocks.append(self._convert_proc_print(block))
            elif upper.startswith("DATA "):
                converted_blocks.append(self._convert_data_step(block))
            else:
                converted_blocks.append(f"# [UNSUPPORTED BLOCK]\n# {block}")

        # Build final output with imports
        header = self._build_imports()
        body = "\n\n".join(converted_blocks)
        return f"{header}\n\n{body}" if header else body

    # ─────────────────────────────────────────────
    # BLOCK SPLITTER
    # ─────────────────────────────────────────────
    def _split_blocks(self, code: str) -> list:
        """Split SAS code into top-level blocks by RUN; or QUIT;"""
        blocks = []
        current = []
        lines = code.splitlines()

        for line in lines:
            stripped = line.strip().upper()
            current.append(line)
            if stripped in ("RUN;", "QUIT;") or stripped.endswith("RUN;") or stripped.endswith("QUIT;"):
                blocks.append("\n".join(current))
                current = []

        if current:
            blocks.append("\n".join(current))

        return blocks

    # ─────────────────────────────────────────────
    # PROC SQL
    # ─────────────────────────────────────────────
    def _convert_proc_sql(self, block: str) -> str:
        lines = block.splitlines()
        # Remove PROC SQL and QUIT lines
        sql_lines = [
            l for l in lines
            if not re.match(r"^\s*(PROC SQL|QUIT)\s*;?\s*$", l, re.IGNORECASE)
        ]
        sql_body = "\n".join(sql_lines).strip().rstrip(";")

        # Extract SELECT statement
        select_match = re.search(
            r"SELECT\s+(.*?)\s+FROM\s+(\S+)(.*?)(?:;|$)",
            sql_body, re.IGNORECASE | re.DOTALL
        )

        if not select_match:
            return f'# PROC SQL\nspark.sql("""\n{sql_body}\n""")'

        select_cols_raw = select_match.group(1).strip()
        from_table      = select_match.group(2).strip().rstrip(";")
        rest            = select_match.group(3).strip()

        # Parse table alias  e.g. employees e  or  employees AS e
        table_name, alias = self._parse_table_alias(from_table)
        df_name = alias or self._table_to_df(table_name)

        lines_out = [f"# PROC SQL → PySpark"]
        lines_out.append(f'{df_name} = spark.table("{table_name}")')

        # JOIN
        join_match = re.search(
            r"(LEFT\s+JOIN|RIGHT\s+JOIN|INNER\s+JOIN|FULL\s+JOIN|JOIN)\s+(\S+)\s+(?:AS\s+)?(\w+)?\s+ON\s+(.+?)(?=WHERE|ORDER|GROUP|HAVING|$)",
            rest, re.IGNORECASE | re.DOTALL
        )
        if join_match:
            join_type   = join_match.group(1).strip().upper()
            join_table  = join_match.group(2).strip()
            join_alias  = join_match.group(3) or self._table_to_df(join_table)
            join_cond   = join_match.group(4).strip().rstrip(";")

            how_map = {
                "LEFT JOIN": "left", "RIGHT JOIN": "right",
                "INNER JOIN": "inner", "FULL JOIN": "outer", "JOIN": "inner"
            }
            how = how_map.get(join_type, "inner")

            lines_out.append(f'{join_alias} = spark.table("{join_table}")')
            lines_out.append(
                f'{df_name} = {df_name}.join({join_alias}, {join_alias}["{self._extract_join_key(join_cond)}"] == {df_name}["{self._extract_join_key(join_cond, right=False)}"], "{how}")'
            )

        # WHERE
        where_match = re.search(r"WHERE\s+(.+?)(?=ORDER|GROUP|HAVING|;|$)", rest, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = self._convert_where(where_match.group(1).strip())
            lines_out.append(f'{df_name} = {df_name}.filter({where_clause})')

        # GROUP BY
        group_match = re.search(r"GROUP\s+BY\s+(.+?)(?=ORDER|HAVING|;|$)", rest, re.IGNORECASE | re.DOTALL)
        if group_match:
            group_cols = [c.strip() for c in group_match.group(1).split(",")]
            group_str  = ", ".join(f'"{c}"' for c in group_cols)
            agg_cols   = self._extract_agg_cols(select_cols_raw)
            if agg_cols:
                self.imports.add("from pyspark.sql import functions as F")
                lines_out.append(f'{df_name} = {df_name}.groupBy({group_str}).agg({agg_cols})')

        # SELECT columns (only if no GROUP BY handled)
        if not group_match:
            if select_cols_raw.strip() != "*":
                cols = self._parse_select_cols(select_cols_raw)
                lines_out.append(f'{df_name} = {df_name}.select({cols})')

        # ORDER BY
        order_match = re.search(r"ORDER\s+BY\s+(.+?)(?=;|$)", rest, re.IGNORECASE | re.DOTALL)
        if order_match:
            order_expr = self._convert_order_by(order_match.group(1).strip(), df_name)
            lines_out.append(order_expr)

        lines_out.append(f'{df_name}.show()')
        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # DATA STEP
    # ─────────────────────────────────────────────
    def _convert_data_step(self, block: str) -> str:
        self.imports.add("from pyspark.sql import functions as F")
        lines = block.splitlines()

        # Extract output dataset name
        data_match = re.match(r"DATA\s+(\S+)\s*;?", lines[0], re.IGNORECASE)
        out_dataset = data_match.group(1) if data_match else "work.output"
        out_df = self._table_to_df(out_dataset)

        # Extract SET dataset
        set_match = re.search(r"SET\s+(\S+)\s*;?", block, re.IGNORECASE)
        in_dataset = set_match.group(1).rstrip(";") if set_match else None
        in_df = self._table_to_df(in_dataset) if in_dataset else "df"

        lines_out = [f"# DATA Step → PySpark"]
        if in_dataset:
            lines_out.append(f'{in_df} = spark.table("{in_dataset}")')

        # WHERE / SUBSET
        where_match = re.search(r"WHERE\s+(.+?);", block, re.IGNORECASE)
        if where_match:
            where_clause = self._convert_where(where_match.group(1).strip())
            lines_out.append(f'{out_df} = {in_df}.filter({where_clause})')
        else:
            lines_out.append(f'{out_df} = {in_df}')

        # Assignment statements  var = expr;
        assign_pattern = re.compile(
            r"^\s*(?!IF|ELSE|WHERE|SET|DATA|RUN|MERGE|BY|IN\b)(\w+)\s*=\s*(.+?)\s*;",
            re.IGNORECASE | re.MULTILINE
        )
        for m in assign_pattern.finditer(block):
            col_name = m.group(1)
            expr     = m.group(2).strip()
            spark_expr = self._convert_expression(expr)
            lines_out.append(f'{out_df} = {out_df}.withColumn("{col_name}", {spark_expr})')

        # IF / THEN / ELSE  (single line)
        if_pattern = re.compile(
            r"IF\s+(.+?)\s+THEN\s+(\w+)\s*=\s*(.+?)(?:\s*ELSE\s+(\w+)\s*=\s*(.+?))?\s*;",
            re.IGNORECASE
        )
        for m in if_pattern.finditer(block):
            condition  = self._convert_where(m.group(1).strip())
            col_name   = m.group(2)
            then_val   = self._convert_expression(m.group(3).strip())
            else_val   = self._convert_expression(m.group(5).strip()) if m.group(5) else "F.lit(None)"
            lines_out.append(
                f'{out_df} = {out_df}.withColumn("{col_name}", F.when({condition}, {then_val}).otherwise({else_val}))'
            )

        # MERGE
        merge_match = re.search(r"MERGE\s+(.+?);", block, re.IGNORECASE | re.DOTALL)
        by_match    = re.search(r"BY\s+(.+?);", block, re.IGNORECASE)
        if merge_match and by_match:
            tables   = re.findall(r"(\w+\.\w+|\w+)(?:\s*\(IN=\w+\))?", merge_match.group(1), re.IGNORECASE)
            by_cols  = [c.strip() for c in by_match.group(1).split()]
            by_str   = ", ".join(f'"{c}"' for c in by_cols)
            if len(tables) >= 2:
                df_a = self._table_to_df(tables[0])
                df_b = self._table_to_df(tables[1])
                lines_out = [f"# MERGE → PySpark join"]
                lines_out.append(f'{df_a} = spark.table("{tables[0]}")')
                lines_out.append(f'{df_b} = spark.table("{tables[1]}")')
                lines_out.append(f'{out_df} = {df_a}.join({df_b}, [{by_str}], "inner")')

        lines_out.append(f'{out_df}.show()')
        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # PROC MEANS / PROC SUMMARY
    # ─────────────────────────────────────────────
    def _convert_proc_means(self, block: str) -> str:
        self.imports.add("from pyspark.sql import functions as F")

        data_match  = re.search(r"DATA\s*=\s*(\S+)", block, re.IGNORECASE)
        class_match = re.search(r"CLASS\s+(.+?);", block, re.IGNORECASE)
        var_match   = re.search(r"VAR\s+(.+?);", block, re.IGNORECASE)
        out_match   = re.search(r"OUT\s*=\s*(\S+)", block, re.IGNORECASE)

        table   = data_match.group(1).rstrip(";") if data_match else "work.data"
        df_name = self._table_to_df(table)

        class_cols = [c.strip() for c in class_match.group(1).split()] if class_match else []
        var_cols   = [c.strip() for c in var_match.group(1).split()]    if var_match   else []
        out_table  = out_match.group(1).rstrip(";")                      if out_match   else None
        out_df     = self._table_to_df(out_table) if out_table else f"{df_name}_summary"

        lines_out = [f"# PROC MEANS → PySpark"]
        lines_out.append(f'{df_name} = spark.table("{table}")')

        agg_exprs = []
        for col in var_cols:
            agg_exprs += [
                f'F.mean("{col}").alias("mean_{col}")',
                f'F.sum("{col}").alias("sum_{col}")',
                f'F.min("{col}").alias("min_{col}")',
                f'F.max("{col}").alias("max_{col}")',
                f'F.count("{col}").alias("n_{col}")',
            ]

        if class_cols:
            group_str = ", ".join(f'"{c}"' for c in class_cols)
            agg_str   = ",\n    ".join(agg_exprs)
            lines_out.append(
                f'{out_df} = {df_name}.groupBy({group_str}).agg(\n    {agg_str}\n)'
            )
        elif agg_exprs:
            agg_str = ",\n    ".join(agg_exprs)
            lines_out.append(f'{out_df} = {df_name}.agg(\n    {agg_str}\n)')
        else:
            lines_out.append(f'{out_df} = {df_name}.describe()')

        lines_out.append(f'{out_df}.show()')
        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # PROC SORT
    # ─────────────────────────────────────────────
    def _convert_proc_sort(self, block: str) -> str:
        self.imports.add("from pyspark.sql import functions as F")

        data_match = re.search(r"DATA\s*=\s*(\S+)", block, re.IGNORECASE)
        out_match  = re.search(r"OUT\s*=\s*(\S+)",  block, re.IGNORECASE)
        by_match   = re.search(r"BY\s+(.+?);",      block, re.IGNORECASE)
        nodup      = bool(re.search(r"NODUPKEY|NODUP", block, re.IGNORECASE))

        table   = data_match.group(1).rstrip(";") if data_match else "work.data"
        df_name = self._table_to_df(table)
        out_df  = self._table_to_df(out_match.group(1).rstrip(";")) if out_match else df_name

        lines_out = [f"# PROC SORT → PySpark"]
        lines_out.append(f'{df_name} = spark.table("{table}")')

        if by_match:
            sort_cols  = by_match.group(1).strip().split()
            sort_exprs = []
            i = 0
            while i < len(sort_cols):
                col = sort_cols[i]
                if col.upper() == "DESCENDING" and i + 1 < len(sort_cols):
                    sort_exprs.append(f'F.col("{sort_cols[i+1]}").desc()')
                    i += 2
                elif col.upper() == "DESCENDING":
                    i += 1
                else:
                    # peek ahead
                    if i + 1 < len(sort_cols) and sort_cols[i+1].upper() == "DESCENDING":
                        sort_exprs.append(f'F.col("{col}").desc()')
                        i += 2
                    else:
                        sort_exprs.append(f'F.col("{col}").asc()')
                        i += 1

            sort_str = ", ".join(sort_exprs)
            lines_out.append(f'{out_df} = {df_name}.orderBy({sort_str})')

            if nodup:
                key_cols = [c for c in sort_cols if c.upper() not in ("DESCENDING", "ASCENDING")]
                key_str  = ", ".join(f'"{c}"' for c in key_cols)
                lines_out.append(f'{out_df} = {out_df}.dropDuplicates([{key_str}])')

        lines_out.append(f'{out_df}.show()')
        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # PROC FREQ
    # ─────────────────────────────────────────────
    def _convert_proc_freq(self, block: str) -> str:
        data_match  = re.search(r"DATA\s*=\s*(\S+)", block, re.IGNORECASE)
        table_match = re.search(r"TABLES\s+(.+?);",  block, re.IGNORECASE)

        table   = data_match.group(1).rstrip(";") if data_match else "work.data"
        df_name = self._table_to_df(table)

        lines_out = [f"# PROC FREQ → PySpark"]
        lines_out.append(f'{df_name} = spark.table("{table}")')

        if table_match:
            cols_raw = table_match.group(1).strip()
            # Handle cross-tabs:  col1 * col2
            if "*" in cols_raw:
                cols = [c.strip() for c in cols_raw.split("*")]
                col_str = ", ".join(f'"{c}"' for c in cols)
                lines_out.append(f'{df_name}.groupBy({col_str}).count().orderBy({col_str}).show()')
            else:
                col = cols_raw.strip()
                lines_out.append(f'{df_name}.groupBy("{col}").count().orderBy(F.col("count").desc()).show()')

        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # PROC PRINT
    # ─────────────────────────────────────────────
    def _convert_proc_print(self, block: str) -> str:
        data_match = re.search(r"DATA\s*=\s*(\S+)", block, re.IGNORECASE)
        var_match  = re.search(r"VAR\s+(.+?);",     block, re.IGNORECASE)
        obs_match  = re.search(r"OBS\s*=\s*(\d+)",  block, re.IGNORECASE)

        table   = data_match.group(1).rstrip(";") if data_match else "work.data"
        df_name = self._table_to_df(table)
        n       = obs_match.group(1) if obs_match else "20"

        lines_out = [f"# PROC PRINT → PySpark"]
        lines_out.append(f'{df_name} = spark.table("{table}")')

        if var_match:
            cols    = [c.strip() for c in var_match.group(1).split()]
            col_str = ", ".join(f'"{c}"' for c in cols)
            lines_out.append(f'{df_name}.select({col_str}).show({n})')
        else:
            lines_out.append(f'{df_name}.show({n})')

        return "\n".join(lines_out)

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def _table_to_df(self, name: str) -> str:
        """work.employees → employees_df"""
        if not name:
            return "df"
        name = name.strip().rstrip(";").strip()
        base = name.split(".")[-1].lower()
        base = re.sub(r"[^a-z0-9_]", "_", base).strip("_")
        return base + "_df"

    def _parse_table_alias(self, raw: str):
        """'employees e' or 'employees AS e' → ('employees', 'e_df')"""
        raw = raw.strip().rstrip(";")
        m = re.match(r"(\S+)\s+(?:AS\s+)?(\w+)$", raw, re.IGNORECASE)
        if m:
            return m.group(1), m.group(2).lower() + "_df"
        return raw, None

    def _extract_join_key(self, condition: str, right: bool = True) -> str:
        """e.dept_id = d.dept_id → dept_id"""
        parts = re.split(r"\s*=\s*", condition)
        if len(parts) == 2:
            part = parts[1] if right else parts[0]
            return part.split(".")[-1].strip()
        return condition

    def _convert_where(self, clause: str) -> str:
        """Convert SAS WHERE clause to PySpark filter expression"""
        clause = clause.strip().rstrip(";")

        # String literals: keep as-is
        # AND / OR
        clause = re.sub(r"\bAND\b", "&", clause, flags=re.IGNORECASE)
        clause = re.sub(r"\bOR\b",  "|", clause, flags=re.IGNORECASE)
        clause = re.sub(r"\bNOT\b", "~", clause, flags=re.IGNORECASE)

        # col = 'value'  →  F.col("col") == 'value'
        clause = re.sub(
            r"(\w+)\s*=\s*('.*?'|\d+\.?\d*)",
            lambda m: f'(F.col("{m.group(1)}") == {m.group(2)})',
            clause
        )
        # col > value
        clause = re.sub(
            r"(\w+)\s*(>=|<=|>|<|!=|<>)\s*('.*?'|\d+\.?\d*)",
            lambda m: f'(F.col("{m.group(1)}") {m.group(2).replace("<>","!=")} {m.group(3)})',
            clause
        )
        # IS NULL / IS NOT NULL
        clause = re.sub(r"(\w+)\s+IS\s+NULL",     r'F.col("\1").isNull()',    clause, flags=re.IGNORECASE)
        clause = re.sub(r"(\w+)\s+IS\s+NOT\s+NULL",r'F.col("\1").isNotNull()', clause, flags=re.IGNORECASE)
        # IN (...)
        clause = re.sub(
            r"(\w+)\s+IN\s+\((.+?)\)",
            lambda m: f'F.col("{m.group(1)}").isin({m.group(2)})',
            clause, flags=re.IGNORECASE
        )

        self.imports.add("from pyspark.sql import functions as F")
        return clause

    def _convert_expression(self, expr: str) -> str:
        """Convert SAS assignment RHS to PySpark expression"""
        expr = expr.strip().strip(";")

        # String literal
        if expr.startswith("'") or expr.startswith('"'):
            return f'F.lit({expr})'

        # Numeric literal
        if re.match(r"^-?\d+\.?\d*$", expr):
            return f'F.lit({expr})'

        # CATX / CAT / CATS
        catx_m = re.match(r"CATX\s*\(\s*'(.+?)'\s*,\s*(.+)\)", expr, re.IGNORECASE)
        if catx_m:
            sep  = catx_m.group(1)
            cols = [c.strip() for c in catx_m.group(2).split(",")]
            col_exprs = ", ".join(f'F.col("{c}")' for c in cols)
            self.imports.add("from pyspark.sql import functions as F")
            return f'F.concat_ws("{sep}", {col_exprs})'

        # UPCASE / LOWCASE
        if re.match(r"UPCASE\s*\((.+)\)", expr, re.IGNORECASE):
            inner = re.match(r"UPCASE\s*\((.+)\)", expr, re.IGNORECASE).group(1)
            return f'F.upper(F.col("{inner.strip()}"))'
        if re.match(r"LOWCASE\s*\((.+)\)", expr, re.IGNORECASE):
            inner = re.match(r"LOWCASE\s*\((.+)\)", expr, re.IGNORECASE).group(1)
            return f'F.lower(F.col("{inner.strip()}"))'

        # SUBSTR
        sub_m = re.match(r"SUBSTR\s*\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", expr, re.IGNORECASE)
        if sub_m:
            return f'F.substring(F.col("{sub_m.group(1)}"), {sub_m.group(2)}, {sub_m.group(3)})'

        # Simple arithmetic:  salary * 0.1
        arith_m = re.match(r"(\w+)\s*([\+\-\*/])\s*(.+)", expr)
        if arith_m:
            col, op, val = arith_m.groups()
            if re.match(r"^-?\d+\.?\d*$", val.strip()):
                return f'F.col("{col}") {op} {val.strip()}'
            return f'F.col("{col}") {op} F.col("{val.strip()}")'

        # Default: treat as column reference
        return f'F.col("{expr}")'

    def _parse_select_cols(self, cols_raw: str) -> str:
        """Parse SELECT columns, handle aliases"""
        cols = [c.strip() for c in cols_raw.split(",")]
        result = []
        for col in cols:
            alias_m = re.match(r"(.+?)\s+(?:AS\s+)?(\w+)$", col, re.IGNORECASE)
            if alias_m and not re.search(r"[(\+\-\*/]", alias_m.group(1)):
                result.append(f'F.col("{alias_m.group(1).strip()}").alias("{alias_m.group(2)}")')
            elif col.strip() == "*":
                return '"*"'
            else:
                result.append(f'"{col}"')
        return ", ".join(result)

    def _extract_agg_cols(self, cols_raw: str) -> str:
        """Extract aggregation functions from SELECT"""
        self.imports.add("from pyspark.sql import functions as F")
        agg_map = {"SUM": "F.sum", "MEAN": "F.mean", "AVG": "F.mean",
                   "MAX": "F.max", "MIN": "F.min", "COUNT": "F.count",
                   "N": "F.count", "STD": "F.stddev"}
        exprs = []
        for col in cols_raw.split(","):
            col = col.strip()
            m = re.match(r"(\w+)\s*\(\s*(\w+)\s*\)(?:\s+(?:AS\s+)?(\w+))?", col, re.IGNORECASE)
            if m:
                fn    = m.group(1).upper()
                field = m.group(2)
                alias = m.group(3) or f"{fn.lower()}_{field}"
                spark_fn = agg_map.get(fn, "F.count")
                exprs.append(f'{spark_fn}("{field}").alias("{alias}")')
        return ", ".join(exprs)

    def _convert_order_by(self, clause: str, df_name: str) -> str:
        """Convert ORDER BY clause"""
        self.imports.add("from pyspark.sql import functions as F")
        parts = [p.strip() for p in clause.split(",")]
        exprs = []
        for part in parts:
            tokens = part.split()
            col    = tokens[0]
            desc   = len(tokens) > 1 and tokens[1].upper() == "DESC"
            exprs.append(f'F.col("{col}").desc()' if desc else f'F.col("{col}").asc()')
        return f'{df_name} = {df_name}.orderBy({", ".join(exprs)})'

    def _build_imports(self) -> str:
        if not self.imports:
            return ""
        return "\n".join(sorted(self.imports))


# ─────────────────────────────────────────────────────────────
# CLI / TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    converter = SASToSparkConverter()

    tests = {
        "PROC SQL with JOIN": """
PROC SQL;
  SELECT e.emp_id, e.name, e.salary, d.dept_name
  FROM work.employees e
  LEFT JOIN work.departments d ON e.dept_id = d.dept_id
  WHERE e.salary > 50000
  ORDER BY e.salary DESC;
QUIT;
""",
        "DATA Step": """
DATA work.filtered;
  SET work.employees;
  WHERE salary > 50000;
  bonus = salary * 0.1;
  full_name = CATX(' ', first_name, last_name);
  IF age >= 30 THEN seniority = 'Senior';
  ELSE seniority = 'Junior';
RUN;
""",
        "PROC MEANS": """
PROC MEANS DATA=work.sales NOPRINT;
  CLASS region product;
  VAR revenue units;
  OUTPUT OUT=work.summary;
RUN;
""",
        "PROC SORT with NODUPKEY": """
PROC SORT DATA=work.customers OUT=work.customers_sorted NODUPKEY;
  BY customer_id DESCENDING purchase_date;
RUN;
""",
        "PROC FREQ": """
PROC FREQ DATA=work.orders;
  TABLES region * product;
RUN;
""",
        "PROC PRINT": """
PROC PRINT DATA=work.employees (OBS=10);
  VAR emp_id name salary department;
RUN;
"""
    }

    for title, sas in tests.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print("SAS INPUT:")
        print(sas.strip())
        print("\nPYSPARK OUTPUT:")
        print(converter.convert(sas))
        print()
