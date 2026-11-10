"""
apply_business_rules.py
=======================

Rule engine for Azure Synapse (PySpark).

Rules live in BUSINESS_RULES.xlsx, sheet "rules". Every RULE_DATA_SELECTION and
RULE_TO_APPLY in that sheet is already a Spark SQL expression, so this module
contains no translation and no parsing: it reads the rules, builds one CASE WHEN
per rule variable, and applies it.

Sheet columns used
------------------
RULE_GROUP           the batch of rules to run (e.g. Investment_Category_Solvency)
RULE_VARIABLE        the column the rule writes
RULE_VARIABLE_TYPE   Character or Numeric - the type the column is cast to
RULE_ID              identifier of the rule, used in error messages
RULE_DATA_SELECTION  Spark SQL condition, e.g. CIC_POS_1_CD in ('3','4')
RULE_TYPE            IF_THEN (write RULE_VARIABLE) or DELETE (drop matching rows)
RULE_TO_APPLY        Spark SQL expression for the value, e.g. 'Shares' or 2
ACTIVE               Y, or N to switch a rule off without deleting it

Order: the rules of a group run in the order they are given, and the LAST
matching rule wins, as in the previous engine. Nothing is sorted here, so the
rules must be loaded in sheet order - load_rules() below does that. To give a
column a default, put a rule with RULE_DATA_SELECTION 1=1 at the top of a group.
"""

from typing import Any, Dict, List, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_TYPES = {"CHARACTER": "string", "NUMERIC": "double"}


def load_rules(path: str, sheet: str = "rules") -> List[Dict[str, Any]]:
    """Read the rules workbook, keeping sheet order."""
    import pandas as pd

    return pd.read_excel(path, sheet_name=sheet).to_dict("records")


def _as_dicts(rules: Any) -> List[Dict[str, Any]]:
    if isinstance(rules, DataFrame):           # a Spark DataFrame does not keep sheet order
        return [r.asDict() for r in rules.collect()]
    if hasattr(rules, "to_dict"):              # pandas
        return rules.to_dict("records")
    return list(rules)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def apply_business_rules(df: DataFrame, rules: Any, group: str) -> DataFrame:
    """Apply every active rule of one rule group to ``df``."""
    wanted = group.strip().lower()
    selected = [
        r for r in _as_dicts(rules)
        if _text(r.get("RULE_GROUP")).lower() == wanted
        and _text(r.get("ACTIVE")).upper() != "N"
    ]
    if not selected:
        raise ValueError(f"No active rules found for rule group {group!r}")

    # DELETE rules first, so the assignments only see the surviving rows
    for rule in selected:
        if _text(rule.get("RULE_TYPE")).upper() == "DELETE":
            df = df.filter(~F.expr(_text(rule["RULE_DATA_SELECTION"])))

    writes = [r for r in selected if _text(r.get("RULE_TYPE")).upper() != "DELETE"]
    for variable in dict.fromkeys(_text(r["RULE_VARIABLE"]) for r in writes):
        variable_rules = [r for r in writes if _text(r["RULE_VARIABLE"]) == variable]

        declared = _text(variable_rules[0].get("RULE_VARIABLE_TYPE")).upper()
        if declared not in _TYPES:
            raise ValueError(f"{variable}: unknown RULE_VARIABLE_TYPE {declared!r}")

        case = None
        for rule in reversed(variable_rules):          # reversed => last match wins
            condition = F.expr(_text(rule["RULE_DATA_SELECTION"]))
            value = F.expr(_text(rule["RULE_TO_APPLY"]))
            case = F.when(condition, value) if case is None else case.when(condition, value)

        unchanged = F.col(variable) if variable in df.columns else F.lit(None)
        df = df.withColumn(variable, case.otherwise(unchanged).cast(_TYPES[declared]))

    return df


def apply_rule_groups(df: DataFrame, rules: Any, groups: Sequence[str]) -> DataFrame:
    """Apply several rule groups in order, each one to the previous result."""
    rule_rows = _as_dicts(rules)
    for group in groups:
        df = apply_business_rules(df, rule_rows, group)
    return df


def rule_groups(rules: Any) -> List[str]:
    """Rule groups present in the sheet, in sheet order."""
    return list(dict.fromkeys(_text(r["RULE_GROUP"]) for r in _as_dicts(rules)))


# ---------------------------------------------------------------------------
# Example usage in a Synapse notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    rules = load_rules("/synfs/ref/.../configuration/01_raw_files/BUSINESS_RULES.xlsx")
    assets_liabilities = spark.table("dbo.Assets_Liabilities_enriched")

    assets_liabilities = apply_rule_groups(assets_liabilities, rules, [
        "Investment_Category_Solvency",
        "Participation_Risk_Strat_NStrat",
        "Infrastructure_type",
        "ALMRiskCards_Category_1",
        "ALMRiskCards_Category_2",
    ])

    assets_liabilities.write.mode("overwrite").saveAsTable("dbo.Assets_Liabilities_enriched")
