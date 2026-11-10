"""
apply_business_rules.py
 
PySpark version of the business rule engine for Azure Synapse.
 
The rules are in BUSINESS_RULES.xlsx, sheet "rules". RULE_DATA_SELECTION and
RULE_TO_APPLY are already Spark SQL, so this code only reads the rules and
applies them - there is no translation and no parsing.
 
Columns used:
    RULE_GROUP           the batch of rules to run
    RULE_VARIABLE        the column the rule writes
    RULE_VARIABLE_TYPE   Character or Numeric
    RULE_DATA_SELECTION  Spark SQL condition
    RULE_TYPE            IF_THEN (write RULE_VARIABLE) or DELETE (drop matching rows)
    RULE_TO_APPLY        Spark SQL expression for the value
    ACTIVE               Y, or N to skip the rule
 
The rules of a group are applied top to bottom, so the last matching rule wins.
"""
 
from pyspark.sql import functions as F
 
 
def apply_business_rules(in_impacted_df,
                         in_business_rule_df,
                         in_business_rule_group,
                         out_table_name=None):
    """Apply all rules of one rule group to a DataFrame."""
 
    rules = in_business_rule_df.filter(
        F.upper(F.trim(F.col("RULE_GROUP"))) == in_business_rule_group.strip().upper()
    ).collect()
 
    if not rules:
        raise ValueError("No rules found for rule group '%s'" % in_business_rule_group)
 
    df = in_impacted_df
 
    for rule in rules:
        if str(rule["ACTIVE"]).strip().upper() == "N":
            continue
 
        condition = str(rule["RULE_DATA_SELECTION"]).strip()
 
        if str(rule["RULE_TYPE"]).strip().upper() == "DELETE":
            df = df.filter("not (%s)" % condition)
            continue
 
        column = str(rule["RULE_VARIABLE"]).strip()
        value = str(rule["RULE_TO_APPLY"]).strip()
 
        if str(rule["RULE_VARIABLE_TYPE"]).strip().upper() == "NUMERIC":
            column_type = "double"
        else:
            column_type = "string"
 
        if column in df.columns:
            current_value = F.col(column)
        else:
            current_value = F.lit(None)
 
        df = df.withColumn(
            column,
            F.when(F.expr(condition), F.expr(value)).otherwise(current_value).cast(column_type)
        )
 
    if out_table_name:
        df.write.mode("overwrite").saveAsTable(out_table_name)
 
    return df
 
 
# ---------------------------------------------------------------------------
# Example usage in a Synapse notebook
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from pyspark.sql import SparkSession
 
    spark = SparkSession.builder.getOrCreate()
 
    rules_df = spark.table("dbo.BUSINESS_RULES")
    assets_liabilities_enriched = spark.table("dbo.Assets_Liabilities_enriched")
 
    rule_groups = [
        "Investment_Category_Solvency",
        "Participation_Risk_Strat_NStrat",
        "Infrastructure_type",
        "ALMRiskCards_Category_1",
        "ALMRiskCards_Category_2",
    ]
 
    for group in rule_groups:
        assets_liabilities_enriched = apply_business_rules(
            in_impacted_df=assets_liabilities_enriched,
            in_business_rule_df=rules_df,
            in_business_rule_group=group,
        )
 
    assets_liabilities_enriched.write.mode("overwrite").saveAsTable(
        "dbo.Assets_Liabilities_enriched"
    )
 
