"""
apply_business_rules.py
=======================

Rule engine for Azure Synapse (PySpark).

Rules live in BUSINESS_RULES.xlsx, sheet "rules". Every CONDITION and VALUE in
that sheet is already a Spark SQL expression, so this module contains no
translation, no parsing and nothing SAS-specific: it reads the rules, builds one
CASE WHEN per target column, and applies it.

Sheet columns used
------------------
RULE_GROUP     the batch of rules to run (e.g. Investment_Category_Solvency)
SEQ            order within the group, lowest first
RULE_ID        unique id, for traceability
ACTION         SET (write TARGET_COLUMN) or DROP_ROW (remove matching rows)
TARGET_COLUMN  column the rule writes
TARGET_TYPE    string or double - the type the column is cast to
CONDITION      Spark SQL boolean expression, e.g. CIC_POS_1_CD in ('3','4')
VALUE          Spark SQL expression, e.g. 'Shares' or 2
ACTIVE         Y, or N to switch a rule off without deleting it

Ordering: within a group the rules run in SEQ order and the LAST matching rule
wins, which is how the original engine behaved. To give a column a default,
put a rule with CONDITION 1=1 at the lowest SEQ.
"""

from typing import Any, Dict, Iterable, List, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_ALLOWED_ACTIONS = {"SET", "DROP_ROW"}
_ALLOWED_TYPES = {"string", "double"}


def _as_dicts(rules: Any) -> List[Dict[str, Any]]:
    """Accept a Spark DataFrame, a pandas DataFrame or a list of dicts."""
    if isinstance(rules, DataFrame):                     # Spark - rule tables are small
        return [r.asDict() for r in rules.collect()]
    if hasattr(rules, "to_dict"):                        # pandas
        return rules.to_dict("records")
    return list(rules)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def apply_business_rules(df: DataFrame, rules: Any, group: str) -> DataFrame:
    """Apply every active rule of one rule group to ``df``."""
    wanted = group.strip().lower()
    selected = [
        r for r in _as_dicts(rules)
        if _clean(r.get("RULE_GROUP")).lower() == wanted
        and _clean(r.get("ACTIVE")).upper() != "N"
    ]
    if not selected:
        raise ValueError(f"No active rules found for rule group {group!r}")
    selected.sort(key=lambda r: float(r["SEQ"]))

    for rule in selected:
        action = _clean(rule.get("ACTION")).upper() or "SET"
        if action not in _ALLOWED_ACTIONS:
            raise ValueError(f"{rule['RULE_ID']}: unknown ACTION {action!r}")

    # 1. DROP_ROW rules first, so later assignments only see surviving rows
    for rule in selected:
        if _clean(rule.get("ACTION")).upper() == "DROP_ROW":
            df = df.filter(~F.expr(_clean(rule["CONDITION"])))

    # 2. SET rules: one CASE WHEN per target column
    set_rules = [r for r in selected if _clean(r.get("ACTION")).upper() != "DROP_ROW"]
    for target in dict.fromkeys(_clean(r["TARGET_COLUMN"]) for r in set_rules):
        target_rules = [r for r in set_rules if _clean(r["TARGET_COLUMN"]) == target]

        target_type = _clean(target_rules[0].get("TARGET_TYPE")).lower() or "string"
        if target_type not in _ALLOWED_TYPES:
            raise ValueError(f"{target}: unknown TARGET_TYPE {target_type!r}")

        case = None
        for rule in reversed(target_rules):             # reversed => last match wins
            condition = F.expr(_clean(rule["CONDITION"]))
            value = F.expr(_clean(rule["VALUE"]))
            case = F.when(condition, value) if case is None else case.when(condition, value)

        unchanged = F.col(target) if target in df.columns else F.lit(None)
        df = df.withColumn(target, case.otherwise(unchanged).cast(target_type))

    return df


def apply_rule_groups(df: DataFrame, rules: Any, groups: Sequence[str]) -> DataFrame:
    """Apply several rule groups in order, each one to the previous result."""
    rule_rows = _as_dicts(rules)
    for group in groups:
        df = apply_business_rules(df, rule_rows, group)
    return df


def rule_groups(rules: Any) -> List[str]:
    """Rule groups present in the sheet, in sheet order."""
    rows = sorted(_as_dicts(rules), key=lambda r: float(r["SEQ"]))
    return list(dict.fromkeys(_clean(r["RULE_GROUP"]) for r in rows))


# ---------------------------------------------------------------------------
# Example usage in a Synapse notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    rules_df = spark.table("dbo.BUSINESS_RULES")                       # the "rules" sheet
    assets_liabilities = spark.table("dbo.Assets_Liabilities_enriched")

    assets_liabilities = apply_rule_groups(assets_liabilities, rules_df, [
        "Investment_Category_Solvency",
        "Participation_Risk_Strat_NStrat",
        "Infrastructure_type",
        "ALMRiskCards_Category_1",
        "ALMRiskCards_Category_2",
    ])

    assets_liabilities.write.mode("overwrite").saveAsTable("dbo.Assets_Liabilities_enriched")
