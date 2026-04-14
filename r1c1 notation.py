def r1c1_to_a1(r1c1_range):
    """
    Convert R1C1 notation to A1 notation
    
    Examples:
        R9C1:R1811C20 -> A9:T1811
        R1C1:R100C5 -> A1:E100
    """
    import re
    
    def col_num_to_letter(n):
        """Convert column number to Excel column letter"""
        result = ""
        while n > 0:
            n -= 1
            result = chr(n % 26 + ord('A')) + result
            n //= 26
        return result
    
    # Parse R1C1 notation
    # Pattern: R<row>C<col>:R<row>C<col>
    pattern = r'R(\d+)C(\d+):R(\d+)C(\d+)'
    match = re.match(pattern, r1c1_range)
    
    if not match:
        raise ValueError(f"Invalid R1C1 notation: {r1c1_range}")
    
    start_row, start_col, end_row, end_col = map(int, match.groups())
    
    # Convert to A1 notation
    start_col_letter = col_num_to_letter(start_col)
    end_col_letter = col_num_to_letter(end_col)
    
    a1_range = f"{start_col_letter}{start_row}:{end_col_letter}{end_row}"
    
    return a1_range


# Usage
r1c1_ref = "R9C1:R1811C20"
a1_ref = r1c1_to_a1(r1c1_ref)
print(f"R1C1: {r1c1_ref}")
print(f"A1:   {a1_ref}")

# Now use in Spark
df = spark.read.format("excel") \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("dataAddress", f"'Sheet1'!{a1_ref}") \
    .load(file_path)

df.show()
